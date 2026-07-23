"""Load the curated eval datasets from JSON into typed, validated case models (todo Task 32).

The golden and red-team datasets ship as JSON under `evals/datasets/` — the same
"curated data as JSON" convention as the guard signatures and `mock_kb/`, so the golden set
can grow from rep corrections (SPEC §4.9) without a code change. Each loader parses one file
into a list of the matching case model; Pydantic validation means a malformed case (a bad
enum value, a missing field) fails loudly at load rather than scoring against a broken case.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from evals.cases import GroundednessCase, RedTeamCase, TriageCase

# The curated datasets live next to this module. Each loader defaults to its file here; the
# path is overridable so tests can point at a fixture (e.g. a deliberately malformed one).
DATASETS_DIR = Path(__file__).resolve().parent / "datasets"


def _load[ModelT: BaseModel](path: Path, model: type[ModelT]) -> list[ModelT]:
    """Parse the JSON array at `path` into a list of validated `model` instances.

    Raises `pydantic.ValidationError` when any element fails the schema and the standard
    JSON/OS errors when the file is missing or malformed — the fail-loud contract the whole
    suite depends on.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return [model.model_validate(item) for item in data]


def load_triage_cases(path: Path | None = None) -> list[TriageCase]:
    """Load the golden triage dataset (message -> expected category/urgency)."""
    return _load(path or DATASETS_DIR / "golden_triage.json", TriageCase)


def load_groundedness_cases(path: Path | None = None) -> list[GroundednessCase]:
    """Load the groundedness dataset (draft + cited sources -> grounded/ungrounded)."""
    return _load(path or DATASETS_DIR / "groundedness.json", GroundednessCase)


def load_redteam_input_cases(path: Path | None = None) -> list[RedTeamCase]:
    """Load the red-team input dataset (prompt-injection attempts the input guard must block)."""
    return _load(path or DATASETS_DIR / "redteam_input.json", RedTeamCase)


def load_redteam_output_cases(path: Path | None = None) -> list[RedTeamCase]:
    """Load the red-team output dataset (forbidden-promise/PII drafts the guard must flag)."""
    return _load(path or DATASETS_DIR / "redteam_output.json", RedTeamCase)
