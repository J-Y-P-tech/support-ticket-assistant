"""Input injection guard: layered prompt-injection screening (plan Task 14 / todo Task 15).

The project's **input gate**. `screen_input` screens customer text and OCR output for
prompt-injection *before* it reaches any LLM node (SPEC §5 step 1, §6), mapping to
OWASP LLM01:2025 (input filtering + content segregation). Prompt injection has no
single reliable filter, so the guard is deliberately layered — two independent checks,
neither trusted alone:

- **Layer 1 — deterministic signatures (always runs, no model call).** Curated regex
  families for the well-known attack shapes: instruction override, system-prompt
  exfiltration, role/mode manipulation, forged chat-role markers, and instruction
  termination. Matched over a whitespace-collapsed copy of the text (case-insensitive)
  so trivial casing/spacing obfuscation does not slip past. This is the reliable floor:
  fast, deterministic, and independent of the very model an attack targets.
- **Layer 2 — optional LLM second opinion (runs only when the rules are clean and an
  `llm` is passed).** A structured-output classifier for novel phrasings the signatures
  miss, validated against `InjectionVerdict` and retried once (mirrors the `validate`
  node). It is a *supplement*: dedicated LLM shields are known to be bypassable (OWASP
  LLM01), so the guard never leans on it as the floor.

The floor is authoritative and **short-circuits**: a signature hit is reported
immediately, without spending an LLM call or exposing the attacker's text to the model
— the second opinion could not change a block into a pass, so running it would add only
latency and needless exposure. The LLM layer runs only when the rules find nothing,
which is precisely the case it exists for. The layers are thus mutually exclusive.
Per the project decision a flagged input is **blocked and routed to a human** — but that
routing is enforced at workflow assembly (todo Task 17); this unit only produces the
verdict. Two guarantees matter, because
this is a security gate rather than a classifier:

- **Fail to the floor, not to an outage.** If the optional LLM layer cannot be parsed
  after its retry budget, the guard degrades to the deterministic result rather than
  flagging every ticket — a flaky classifier must not become a self-inflicted denial of
  service, and the deterministic floor still protects. (The reliable layer is the
  deterministic one; the bypassable layer is the one allowed to fail open.)
- **Cheap short-circuit.** Empty/whitespace-only text is clean without a model call.

This is a plain async function, independent of LangGraph; the workflow adapter that
wires it in — and blocks-to-human on a flag — is added in Task 17. `max_attempts` is a
tuning knob supplied by that adapter (like the nodes'), not a constant here. The
classifier prompt lives in-repo via the prompt registry; Langfuse-managed resolution is
deferred to Task 28.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import ValidationError

from app.llm.base import LLM
from app.prompts.registry import get_prompt
from app.schemas.guardrails import InjectionScreenResult, InjectionVerdict

# Grab the first `{`-to-last-`}` span as the JSON object, tolerating any prose or
# code-fence framing the classifier may wrap around it (mirrors the other nodes).
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

# Deterministic signature families (Layer 1) live in a JSON data file next to this
# module, not hardcoded here, so security/ops can add or tune signatures — a moving
# target as new attack phrasings appear — by editing data and shipping it, without
# touching this logic (the same "curated data as JSON" convention as `mock_kb/`, and
# the same load-from-a-seam idea as the prompt registry). The file is loaded and its
# patterns compiled once at import. These cover the textbook direct-injection shapes
# (OWASP LLM01); the LLM layer covers novel phrasings a fixed list cannot enumerate.
# Known limitation: per-character obfuscation ("i g n o r e") survives whitespace-
# collapse — that is what Layer 2 is for.
_SIGNATURES_PATH = Path(__file__).resolve().parent / "injection_signatures.json"


def _load_signatures(path: Path) -> dict[str, list[re.Pattern[str]]]:
    """Load the signature families from `path` and compile each pattern (case-insensitive).

    Returns a category -> compiled-patterns map — the deterministic layer's rule set.
    Adding a family or pattern to the JSON is picked up here at startup with no code
    change. A malformed regex raises `re.error` at load time (fail loud, not at runtime),
    so a bad edit is caught immediately rather than silently disabling a signature.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        family["category"]: [re.compile(pattern, re.IGNORECASE) for pattern in family["patterns"]]
        for family in data["families"]
    }


# The compiled rule set, loaded once from the data file at import.
_RULES: dict[str, list[re.Pattern[str]]] = _load_signatures(_SIGNATURES_PATH)


def _normalize(text: str) -> str:
    """Collapse whitespace runs to single spaces (and strip) for stable signature matching.

    Casing is left to the case-insensitive patterns, so the matched snippet kept as
    evidence stays readable in the original case. This neutralizes padded-whitespace and
    line-break obfuscation; per-character spacing is out of scope (see the module note).
    """
    return re.sub(r"\s+", " ", text).strip()


def _dedupe(items: list[str]) -> list[str]:
    """Return `items` with duplicates removed, preserving first-seen order.

    Used to merge the two layers' categories and evidence so a family both layers
    reported is listed once.
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

    For each family, every pattern match contributes its matched snippet to the
    evidence; a family that matches at all is named once in the categories. Both lists
    are empty when nothing trips — the clean, common case.
    """
    categories: list[str] = []
    evidence: list[str] = []
    for category, patterns in _RULES.items():
        hits = [match.group(0) for pattern in patterns for match in pattern.finditer(normalized)]
        if hits:
            categories.append(category)
            evidence.extend(hits)
    return categories, evidence


def _parse(raw: str) -> InjectionVerdict:
    """Parse one classifier response into an `InjectionVerdict`, raising on any invalidity.

    Raises `ValueError` when no JSON object is present or it is malformed
    (`json.JSONDecodeError` is a `ValueError`) and `ValidationError` when a field has the
    wrong type. The caller treats either as a failed attempt.
    """
    match = _JSON_OBJECT_RE.search(raw)
    if match is None:
        raise ValueError("no JSON object found in classifier output")
    data = json.loads(match.group(0))
    return InjectionVerdict.model_validate(data)


async def _classify(text: str, llm: LLM, *, max_attempts: int) -> InjectionVerdict | None:
    """Ask the model whether `text` is an injection attempt, retrying on bad output.

    Sends the in-repo guard prompt (with `think=True`) and parses the reply, retrying
    until the attempt budget is spent. Returns the validated verdict, or `None` when
    every attempt failed — the caller degrades to the deterministic floor on `None`
    rather than guessing a verdict.
    """
    prompt = get_prompt("input_guard").format(text=text)
    for attempt in range(1, max_attempts + 1):
        raw = await llm.generate(prompt, think=True)
        try:
            return _parse(raw)
        except (ValueError, ValidationError):
            if attempt == max_attempts:
                return None
    return None


async def screen_input(
    text: str, llm: LLM | None = None, *, max_attempts: int = 1
) -> InjectionScreenResult:
    """Screen `text` for prompt-injection, deterministic floor first, LLM only if clean.

    Always runs the deterministic signature floor. The floor is authoritative and
    **short-circuits**: when it already catches an attack the outcome is a block
    regardless of any second opinion, so the guard reports immediately and does *not*
    call the model — that spares a needless round-trip and avoids feeding the attacker's
    text into the very LLM it targets. Only when the rules are clean (and an `llm` is
    supplied) does the LLM second-opinion classifier run, with the `max_attempts` retry
    budget the workflow adapter passes from config — that is exactly the case it exists
    for: novel phrasings the signatures miss. The layers are therefore mutually
    exclusive, and `detector` records which one fired: `"rules"`, `"llm"`, or `"none"`.

    Two safety behaviours: whitespace-only text short-circuits to a clean verdict with no
    model call, and a classifier that never returns valid output degrades to the
    deterministic (clean) result rather than flagging everything — the reliable floor
    still holds. Flagging is a verdict, not an action: the block-and-route-to-human is
    applied when the workflow is assembled (todo Task 17).
    """
    if not text.strip():
        return InjectionScreenResult(flagged=False, detector="none")

    rule_categories, rule_evidence = _screen_rules(_normalize(text))
    if rule_categories:
        # The floor caught it — block without consulting (or exposing text to) the model.
        return InjectionScreenResult(
            flagged=True,
            categories=rule_categories,
            evidence=rule_evidence,
            detector="rules",
        )

    if llm is not None:
        verdict = await _classify(text, llm, max_attempts=max_attempts)
        if verdict is not None and verdict.is_injection:
            return InjectionScreenResult(
                flagged=True,
                categories=_dedupe(verdict.categories),
                evidence=_dedupe(verdict.evidence),
                detector="llm",
            )

    return InjectionScreenResult(flagged=False, detector="none")
