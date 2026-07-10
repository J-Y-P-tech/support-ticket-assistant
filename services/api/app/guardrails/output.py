"""Output guard: layered forbidden-promise / PII / tone screening (plan Task 15 / todo Task 16).

The project's **output gate**. `screen_output` screens the drafted reply *before* it
reaches the rep (SPEC §4.6), mapping to OWASP LLM06:2025 (sensitive-information
disclosure) and LLM05 (improper output handling). It mirrors the input guard's layered,
short-circuiting shape, but its two layers cover complementary concerns:

- **Layer 1 — deterministic signatures (always runs, no model call).** Curated regex
  families for the *objective*, reliably-detectable violations: forbidden financial/legal
  commitments (guarantees, refund/reimbursement promises, accepting liability) and PII
  leakage (card/account numbers, SSNs). Matched over a whitespace-collapsed copy of the
  draft (case-insensitive) so trivial spacing/casing variation does not slip past. This
  is the reliable floor: fast, deterministic, independent of the model that wrote the draft.
- **Layer 2 — optional LLM second opinion (runs only when the rules are clean and an
  `llm` is passed).** A structured-output classifier for the *subjective* check the
  signatures cannot make — tone (rude, dismissive, condescending, blaming) — validated
  against `ToneVerdict` and retried once (mirrors the `validate` node and the input guard).

The floor is authoritative and **short-circuits**: a signature hit is reported
immediately, without spending an LLM call — the tone opinion could not change a
promise/PII flag into a pass, so running it would add only latency. The LLM layer runs
only when the rules find nothing, which is precisely the case it exists for. The layers
are thus mutually exclusive, and `detector` records which one fired: `"rules"`, `"llm"`,
or `"none"`. Per the project decision a flagged draft is **surfaced to the rep with
warnings** (never discarded) — but that routing is enforced at workflow assembly (todo
Task 17); this unit only produces the verdict. Two guarantees matter, because this is a
safety gate rather than a classifier:

- **Fail to the floor, not to an outage.** If the optional LLM layer cannot be parsed
  after its retry budget, the guard degrades to the deterministic result rather than
  flagging every draft — a flaky classifier must not become a self-inflicted denial of
  service, and the deterministic floor still protects.
- **Never emit raw PII.** A PII family's evidence is **masked** to the last four digits,
  so the verdict that flows to the rep and the audit trail never carries a full
  account/card number (SPEC §5/§7 — never log full account/card numbers).

This is a plain async function, independent of LangGraph; the workflow adapter that wires
it in — and surfaces-to-human on a flag — is added in Task 17. `max_attempts` is a tuning
knob supplied by that adapter (like the nodes'), not a constant here. The tone-classifier
prompt lives in-repo via the prompt registry; Langfuse-managed resolution is deferred to
Task 28.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from app.llm.base import LLM
from app.prompts.registry import get_prompt
from app.schemas.guardrails import OutputScreenResult, ToneVerdict

# Grab the first `{`-to-last-`}` span as the JSON object, tolerating any prose or
# code-fence framing the classifier may wrap around it (mirrors the other nodes).
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

# Deterministic signature families (Layer 1) live in a JSON data file next to this
# module, not hardcoded here, so security/ops can add or tune signatures — a moving
# target as new phrasings appear — by editing data and shipping it, without touching this
# logic (the same "curated data as JSON" convention as the input guard and `mock_kb/`).
# The file is loaded and its patterns compiled once at import. A family may set
# `redact: true`, meaning its matched snippets are masked to the last four digits before
# being recorded as evidence — the mechanism that keeps a PII leak out of the verdict.
_SIGNATURES_PATH = Path(__file__).resolve().parent / "output_signatures.json"

# How many trailing digits a masked (PII) evidence snippet keeps, so the rep can still
# identify the leak without the verdict carrying the full number.
_MASK_KEEP = 4


@dataclass(frozen=True)
class _Family:
    """One compiled signature family: its label, its patterns, and whether to mask evidence.

    `redact` marks a family whose matched snippets are PII (card/account numbers, SSNs);
    the screener masks those to the last four digits before recording them as evidence.
    """

    category: str
    patterns: tuple[re.Pattern[str], ...]
    redact: bool


def _load_signatures(path: Path) -> list[_Family]:
    """Load the signature families from `path` and compile each pattern (case-insensitive).

    Returns the ordered list of families — the deterministic layer's rule set. Adding a
    family or pattern (or a `redact` flag) to the JSON is picked up here at startup with no
    code change. A malformed regex raises `re.error` at load time (fail loud, not at
    runtime), so a bad edit is caught immediately rather than silently disabling a signature.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        _Family(
            category=family["category"],
            patterns=tuple(re.compile(pattern, re.IGNORECASE) for pattern in family["patterns"]),
            redact=bool(family.get("redact", False)),
        )
        for family in data["families"]
    ]


# The compiled rule set, loaded once from the data file at import.
_RULES: list[_Family] = _load_signatures(_SIGNATURES_PATH)


def _normalize(text: str) -> str:
    """Collapse whitespace runs to single spaces (and strip) for stable signature matching.

    Casing is left to the case-insensitive patterns, so the matched snippet kept as
    evidence stays readable in the original case. This neutralizes padded-whitespace and
    line-break obfuscation of a forbidden promise, and keeps grouped card numbers matchable.
    """
    return re.sub(r"\s+", " ", text).strip()


def _mask_digits(snippet: str) -> str:
    """Mask all but the last four digits of `snippet`, preserving its non-digit characters.

    Turns a matched PII snippet into a leak-free form for evidence: "4111 2222 3333 4455"
    becomes "**** **** **** 4455" and "123-45-6789" becomes "***-**-6789". Separators are
    kept so the rep sees the shape; only the digits are hidden. When there are four or fewer
    digits nothing is masked (there is no leak to hide).
    """
    total = sum(character.isdigit() for character in snippet)
    seen = 0
    out: list[str] = []
    for character in snippet:
        if character.isdigit():
            seen += 1
            out.append(character if seen > total - _MASK_KEEP else "*")
        else:
            out.append(character)
    return "".join(out)


def _dedupe(items: list[str]) -> list[str]:
    """Return `items` with duplicates removed, preserving first-seen order.

    Used to tidy the LLM layer's categories and evidence so a tone problem the classifier
    names twice is listed once.
    """
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _screen_rules(normalized: str) -> tuple[list[str], list[str]]:
    """Run the deterministic signature layer, returning (categories, evidence).

    For each family, every pattern match contributes its matched snippet to the evidence
    (masked to the last four digits when the family is `redact`); a family that matches at
    all is named once in the categories. Both lists are empty when nothing trips — the
    clean, common case.
    """
    categories: list[str] = []
    evidence: list[str] = []
    for family in _RULES:
        hits = [
            match.group(0) for pattern in family.patterns for match in pattern.finditer(normalized)
        ]
        if hits:
            categories.append(family.category)
            evidence.extend(_mask_digits(hit) if family.redact else hit for hit in hits)
    return categories, evidence


def _parse(raw: str) -> ToneVerdict:
    """Parse one tone-classifier response into a `ToneVerdict`, raising on any invalidity.

    Raises `ValueError` when no JSON object is present or it is malformed
    (`json.JSONDecodeError` is a `ValueError`) and `ValidationError` when a field has the
    wrong type. The caller treats either as a failed attempt.
    """
    match = _JSON_OBJECT_RE.search(raw)
    if match is None:
        raise ValueError("no JSON object found in classifier output")
    data = json.loads(match.group(0))
    return ToneVerdict.model_validate(data)


async def _classify_tone(draft: str, llm: LLM, *, max_attempts: int) -> ToneVerdict | None:
    """Ask the model whether `draft` has a tone problem, retrying on bad output.

    Sends the in-repo tone-guard prompt (with `think=True`) and parses the reply, retrying
    until the attempt budget is spent. Returns the validated verdict, or `None` when every
    attempt failed — the caller degrades to the deterministic floor on `None` rather than
    guessing a verdict.
    """
    prompt = get_prompt("output_guard").format(draft=draft)
    for attempt in range(1, max_attempts + 1):
        raw = await llm.generate(prompt, think=True)
        try:
            return _parse(raw)
        except (ValueError, ValidationError):
            if attempt == max_attempts:
                return None
    return None


async def screen_output(
    draft: str, llm: LLM | None = None, *, max_attempts: int = 1
) -> OutputScreenResult:
    """Screen `draft` for violations, deterministic floor first, tone check only if clean.

    Always runs the deterministic signature floor (forbidden promises + PII). The floor is
    authoritative and **short-circuits**: when it already catches a violation the outcome is
    a flag regardless of any tone opinion, so the guard reports immediately and does *not*
    call the model — that spares a needless round-trip. Only when the rules are clean (and
    an `llm` is supplied) does the LLM tone classifier run, with the `max_attempts` retry
    budget the workflow adapter passes from config — that is exactly the case it exists for:
    the subjective judgement the signatures cannot make. The layers are therefore mutually
    exclusive, and `detector` records which one fired: `"rules"`, `"llm"`, or `"none"`.

    Two safety behaviours: whitespace-only text short-circuits to a clean verdict with no
    model call, and a tone classifier that never returns valid output degrades to the
    deterministic (clean) result rather than flagging everything — the reliable floor still
    holds. PII evidence is masked to the last four digits, so a flag never carries a raw
    account/card number. Flagging is a verdict, not an action: the surface-to-human is
    applied when the workflow is assembled (todo Task 17).
    """
    if not draft.strip():
        return OutputScreenResult(flagged=False, detector="none")

    rule_categories, rule_evidence = _screen_rules(_normalize(draft))
    if rule_categories:
        # The floor caught it — flag without consulting the model.
        return OutputScreenResult(
            flagged=True,
            categories=rule_categories,
            evidence=rule_evidence,
            detector="rules",
        )

    if llm is not None:
        verdict = await _classify_tone(draft, llm, max_attempts=max_attempts)
        if verdict is not None and verdict.has_violation:
            return OutputScreenResult(
                flagged=True,
                categories=_dedupe(verdict.categories),
                evidence=_dedupe(verdict.evidence),
                detector="llm",
            )

    return OutputScreenResult(flagged=False, detector="none")
