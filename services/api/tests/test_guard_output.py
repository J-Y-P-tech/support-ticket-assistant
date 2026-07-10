"""Unit tests for the output guard (plan Task 15 / todo Task 16).

`screen_output` is the project's **output gate**: it screens the drafted reply for
forbidden financial/legal commitments, PII leakage, and tone violations *before* the
draft reaches the rep (SPEC §4.6; OWASP LLM06 sensitive-information disclosure / LLM05
improper output handling). It mirrors the input guard's layered, short-circuiting
shape, and the layers are independent:

- **Layer 1 — deterministic signatures (always runs, no model call).** Curated regex
  families for the objective, reliably-detectable violations: forbidden promises
  (guarantees, refund/reimbursement commitments) and PII leakage (card/account numbers,
  SSNs), matched over a whitespace-collapsed copy of the draft so trivial spacing/casing
  variation does not slip past. This is the reliable floor.
- **Layer 2 — optional LLM second opinion (runs only when the rules are clean and an
  `llm` is passed).** A structured-output classifier for the *subjective* check the
  signatures cannot make — tone (rude, dismissive, condescending, blaming) — validated
  against `ToneVerdict` and retried once, mirroring the `validate` node and the input
  guard's LLM layer.

The deterministic floor is authoritative and **short-circuits**: a signature hit is
reported immediately, without spending an LLM call — the tone opinion could not change a
promise/PII block into a pass, so running it would add only latency. The LLM layer runs
only when the rules are clean, which is precisely the case it exists for. The layers are
thus mutually exclusive, and `detector` records which one fired.

Two behaviours differ from the classifier nodes, because this is a safety gate:

- **Fail to the floor, not to an outage.** If the optional LLM layer cannot be parsed
  after its retry budget, the guard degrades to the deterministic result rather than
  flagging every draft — a flaky classifier must not become a self-inflicted denial of
  service. The deterministic floor still protects.
- **Never emit raw PII.** The PII layer's evidence is **masked** (last four digits only),
  so the verdict that flows to the rep and the audit trail never carries a full
  account/card number (SPEC §5/§7 — never log full account/card numbers).

Per the project decision a flagged draft is **surfaced to the rep with warnings**, not
discarded — but that routing is enforced at workflow assembly (todo Task 17); this unit
only produces the verdict. The tests here therefore pin the *verdict*, not the routing.
Layer 2 is driven with a deterministic `FakeLLM`, never the host model (SPEC §10/§12).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.guardrails import output
from app.guardrails.output import (
    _RULES,
    _SIGNATURES_PATH,
    _load_signatures,
    screen_output,
)
from app.llm.fake import FakeLLM
from app.schemas.guardrails import OutputScreenResult

# The retry budget the tests pass explicitly (production supplies it from config).
_MAX = 2


def _tone(
    has_violation: bool, categories: list[str] | None = None, evidence: list[str] | None = None
) -> str:
    """Render an LLM tone-classifier response as the JSON the model is asked to emit."""
    return json.dumps(
        {
            "has_violation": has_violation,
            "categories": categories or [],
            "evidence": evidence or [],
        }
    )


# --- Layer 1: forbidden promises (deterministic floor) -----------------------------


async def test_clean_draft_is_not_flagged_by_the_rules_alone() -> None:
    """An ordinary grounded reply trips no signature; rules-only mode returns a clean verdict.

    With no `llm`, only the deterministic floor runs. A benign draft must pass with
    `flagged=False`, no categories, and a `detector` recording that nothing caught it.
    """
    draft = (
        "Thank you for reaching out. To update your mailing address, sign in and open "
        "Profile > Contact details. A representative will follow up if anything is unclear."
    )

    outcome = await screen_output(draft)

    assert isinstance(outcome, OutputScreenResult)
    assert outcome.flagged is False
    assert outcome.categories == []
    assert outcome.evidence == []
    assert outcome.detector == "none"


async def test_guarantee_is_flagged_by_the_rules() -> None:
    """A guarantee is a forbidden commitment the deterministic floor catches (acceptance).

    "We guarantee ..." is exactly the financial/legal commitment the guard must block
    without a human (SPEC §4.6). The rules must flag it as `forbidden_promise` and surface
    the offending snippet as evidence, with no LLM call needed.
    """
    draft = "We guarantee your account will never be charged a fee again."

    outcome = await screen_output(draft)

    assert outcome.flagged is True
    assert "forbidden_promise" in outcome.categories
    assert outcome.detector == "rules"
    assert any("guarantee" in snippet.lower() for snippet in outcome.evidence)


async def test_refund_commitment_is_flagged_by_the_rules() -> None:
    """An unconditional refund commitment is a forbidden financial promise.

    "We will refund you ..." commits the institution to a payout without a human decision;
    the floor must flag it as `forbidden_promise` (SPEC §4.6).
    """
    outcome = await screen_output("Don't worry — we will refund you the full amount today.")

    assert outcome.flagged is True
    assert "forbidden_promise" in outcome.categories
    assert outcome.detector == "rules"


async def test_case_and_whitespace_obfuscation_still_trips_the_promise_rules() -> None:
    """Casing and padded whitespace do not evade the floor (normalization before match).

    The rules match a whitespace-collapsed, case-insensitive copy of the draft, so a
    shouted, spaced-out variant of a forbidden promise is caught just like the plain form.
    """
    outcome = await screen_output("WE   GUARANTEE   a full refund, no questions asked.")

    assert outcome.flagged is True
    assert "forbidden_promise" in outcome.categories


async def test_rules_only_mode_needs_no_model() -> None:
    """Passing no `llm` runs the deterministic floor alone and still flags a forbidden promise."""
    outcome = await screen_output("I promise this will be fixed by tomorrow.", llm=None)

    assert outcome.flagged is True
    assert "forbidden_promise" in outcome.categories
    assert outcome.detector == "rules"


# --- Layer 1: PII leakage (deterministic floor, masked evidence) -------------------


async def test_card_number_leak_is_flagged_and_evidence_is_masked() -> None:
    """A full card number in the draft is flagged as a PII leak with masked evidence.

    The floor catches the leak (`pii_leak`), but — per SPEC §5/§7, never log a full
    account/card number — the evidence it records is masked to the last four digits, so
    the verdict that flows to the rep and audit trail never carries the raw number.
    """
    outcome = await screen_output("Your card 4111 2222 3333 4455 has been updated.")

    assert outcome.flagged is True
    assert "pii_leak" in outcome.categories
    assert outcome.detector == "rules"

    joined = " ".join(outcome.evidence)
    assert "4455" in joined  # last four preserved so the rep can identify the leak
    assert "4111" not in joined  # leading digits masked
    assert "4111222233334455" not in joined  # the raw number never appears, spaced or not


async def test_ssn_leak_is_flagged_and_masked() -> None:
    """A Social Security number in the draft is flagged and masked to the last four digits."""
    outcome = await screen_output("For reference your SSN 123-45-6789 is on file.")

    assert outcome.flagged is True
    assert "pii_leak" in outcome.categories

    joined = " ".join(outcome.evidence)
    assert "6789" in joined
    assert "123-45-6789" not in joined
    assert "123" not in joined


async def test_masked_last_four_reference_is_not_flagged() -> None:
    """A legitimate "ending in 1234" reference is not a leak and must not be flagged.

    Drafts routinely refer to an account by its last four digits; that is the safe,
    non-leaking form and must pass the PII floor.
    """
    outcome = await screen_output("We've updated the card ending in 4455 on your profile.")

    assert outcome.flagged is False
    assert outcome.detector == "none"


# --- Layer 2: tone (optional LLM second opinion) -----------------------------------


async def test_llm_layer_catches_tone_the_rules_cannot() -> None:
    """A rude but promise-free, PII-free draft is caught by the LLM tone check.

    Tone is the subjective judgement the signatures cannot make. The draft trips no rule,
    but the classifier flags its tone. The verdict must be flagged, carry the classifier's
    category and evidence, and record `detector="llm"`.
    """
    llm = FakeLLM(_tone(True, ["blaming"], ["your own fault"]))

    outcome = await screen_output(
        "Honestly, this is your own fault for not reading the terms.",
        llm=llm,
        max_attempts=_MAX,
    )

    assert outcome.flagged is True
    assert "blaming" in outcome.categories
    assert any("fault" in snippet for snippet in outcome.evidence)
    assert outcome.detector == "llm"


async def test_rule_hit_short_circuits_and_never_calls_the_model() -> None:
    """A signature hit reports immediately without consulting the tone classifier.

    The deterministic floor is authoritative: once it catches a promise or PII leak the
    outcome is a flag regardless of tone, so the guard must not spend an LLM call. Detector
    is `rules` and the classifier is never invoked, even though one was supplied and would
    have flagged.
    """
    llm = FakeLLM(_tone(True, ["rude"], ["some rude phrase"]))

    outcome = await screen_output(
        "We guarantee a full refund within 24 hours.",
        llm=llm,
        max_attempts=_MAX,
    )

    assert outcome.flagged is True
    assert outcome.detector == "rules"
    assert "forbidden_promise" in outcome.categories
    assert "rude" not in outcome.categories  # the tone classifier was never consulted
    assert llm.calls == []  # short-circuited before any model call


async def test_rules_flag_even_when_the_llm_says_tone_is_fine() -> None:
    """The floor wins: a promise is flagged even if the tone classifier clears the draft.

    A permissive tone verdict cannot launder a forbidden promise back to clean; the
    deterministic layer is the reliable protection. Detector is `rules`.
    """
    llm = FakeLLM(_tone(False))

    outcome = await screen_output("We guarantee approval of your loan.", llm=llm, max_attempts=_MAX)

    assert outcome.flagged is True
    assert "forbidden_promise" in outcome.categories
    assert outcome.detector == "rules"


async def test_clean_draft_passes_when_both_layers_agree_it_is_clean() -> None:
    """A benign draft that neither layer flags returns a clean verdict with `detector="none"`."""
    llm = FakeLLM(_tone(False))

    outcome = await screen_output(
        "Happy to help — you can reset your password from the sign-in screen.",
        llm=llm,
        max_attempts=_MAX,
    )

    assert outcome.flagged is False
    assert outcome.categories == []
    assert outcome.detector == "none"


async def test_llm_is_shown_the_screened_draft() -> None:
    """The tone-classifier prompt carries the exact draft under screening, so it judges it."""
    llm = FakeLLM(_tone(False))
    draft = "UNIQUE-DRAFT-SENTINEL your request has been received."

    await screen_output(draft, llm=llm, max_attempts=_MAX)

    assert draft in llm.calls[0]["prompt"]


async def test_invalid_then_valid_llm_output_is_retried() -> None:
    """A tone response that will not parse is retried once, then the valid one wins."""
    llm = FakeLLM(["not json at all", _tone(True, ["dismissive"], ["whatever"])])

    outcome = await screen_output(
        "A polite-looking but subtly dismissive reply.", llm=llm, max_attempts=_MAX
    )

    assert outcome.flagged is True
    assert "dismissive" in outcome.categories
    assert len(llm.calls) == 2  # first attempt failed, second succeeded


async def test_unparseable_llm_degrades_to_the_deterministic_floor() -> None:
    """A broken tone classifier degrades to the rules result rather than flagging everything.

    Fail-safe direction: if the LLM layer never returns valid output, the guard does NOT
    manufacture a tone violation that would flag every draft — it returns the deterministic
    floor's result. Here the rules are clean, so the draft passes. The guard must not raise.
    """
    llm = FakeLLM(["garbage", "still garbage"])

    outcome = await screen_output(
        "You can update your address from the profile page.", llm=llm, max_attempts=_MAX
    )

    assert outcome.flagged is False
    assert outcome.detector == "none"


async def test_broken_llm_still_reports_a_rules_hit() -> None:
    """Even with a broken tone classifier, a signature match is still flagged (floor holds).

    The classifier fails, but the deterministic layer catches the promise, so the draft is
    flagged on the rules alone — the LLM outage never weakens the floor.
    """
    llm = FakeLLM(["garbage", "still garbage"])

    outcome = await screen_output("We guarantee a refund.", llm=llm, max_attempts=_MAX)

    assert outcome.flagged is True
    assert "forbidden_promise" in outcome.categories
    assert outcome.detector == "rules"


async def test_empty_draft_is_clean_without_calling_the_model() -> None:
    """Whitespace-only text short-circuits to a clean verdict, spending no model call."""
    llm = FakeLLM(_tone(True))  # would flag if consulted

    outcome = await screen_output("   \n  ", llm=llm, max_attempts=_MAX)

    assert outcome.flagged is False
    assert outcome.detector == "none"
    assert llm.calls == []  # the classifier was never invoked


# --- Externalized signatures: the deterministic rules live in an editable JSON data
# --- file the module loads at startup, so security/ops can add or tune them without a
# --- code change. These tests pin that seam (load, take-effect, fail-loud).


def test_signatures_load_from_the_shipped_data_file() -> None:
    """The shipped signatures file loads and compiles into the expected categories.

    Externalizing the rules to JSON must not lose a family: loading the data file yields
    at least one compiled pattern for every violation category the floor relies on.
    """
    rules = _load_signatures(_SIGNATURES_PATH)
    categories = {family.category for family in rules}
    expected = {"forbidden_promise", "pii_leak"}

    assert expected <= categories
    assert all(
        family.patterns and all(isinstance(pattern, re.Pattern) for pattern in family.patterns)
        for family in rules
    )


def test_the_shipped_file_is_the_active_rule_set() -> None:
    """The guard's active rules are exactly what the data file compiles to (loaded at import).

    Pins that `_RULES` is sourced from the JSON file rather than a stale in-code copy, so
    an edit to the file is what the guard actually runs.
    """
    active = {family.category for family in _RULES}
    shipped = {family.category for family in _load_signatures(_SIGNATURES_PATH)}

    assert active == shipped


async def test_a_new_signature_in_the_file_takes_effect_without_code_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adding a signature to the data file changes detection with no code edit.

    The whole point of externalizing the rules: a phrase the shipped signatures ignore
    becomes flagged — with the new category — once a family for it is added to the JSON and
    loaded. Security/ops can extend the deterministic floor by editing data alone.
    """
    novel = "we hereby waive all your outstanding obligations forever"
    assert (await screen_output(novel)).flagged is False  # the shipped rules do not know it

    custom = tmp_path / "output_signatures.json"
    custom.write_text(
        json.dumps(
            {
                "families": [
                    {
                        "category": "custom_waiver",
                        "description": "a signature added by ops",
                        "patterns": [r"waive\s+all\s+your\s+outstanding\s+obligations"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(output, "_RULES", _load_signatures(custom))

    outcome = await screen_output(novel)

    assert outcome.flagged is True
    assert "custom_waiver" in outcome.categories
    assert outcome.detector == "rules"


def test_a_malformed_regex_in_the_file_fails_loud(tmp_path: Path) -> None:
    """A broken pattern in the data file raises at load, never silently disabling a signature.

    A bad edit must fail immediately (`re.error`) so it is caught before deploy, rather than
    being swallowed into a signature that quietly matches nothing.
    """
    bad = tmp_path / "output_signatures.json"
    bad.write_text(
        json.dumps(
            {"families": [{"category": "broken", "description": "x", "patterns": ["(unclosed"]}]}
        ),
        encoding="utf-8",
    )

    with pytest.raises(re.error):
        _load_signatures(bad)
