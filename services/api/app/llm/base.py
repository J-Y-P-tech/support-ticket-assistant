"""Shared LLM interface (plan Task 9 / todo Task 10).

Every model-using node (triage, retrieval-grounded drafting, OCR/extraction)
depends on this small async interface rather than on a concrete backend. In
production the implementation is `OllamaLLM` (the single multimodal host model,
SPEC §4.2); in tests and CI it is the deterministic `FakeLLM`, so no path ever
reaches the network (SPEC §10/§12). Keeping the contract this narrow is what lets
the two swap freely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLM(ABC):
    """A swappable large-language-model backend."""

    @abstractmethod
    async def generate(
        self, prompt: str, *, images: list[str] | None = None, think: bool = True
    ) -> str:
        """Return the model's final-answer text for `prompt`.

        `images` carries optional base64-encoded attachments for the multimodal
        (vision/OCR) path; text-only callers omit it. `think` asks the model to
        reason first (the backend keeps that trace out of the returned answer —
        SPEC Appendix A); it defaults **on** for answer quality, and a caller can
        pass False to trade reasoning for speed. The return value is the final
        answer, already free of any reasoning trace.
        """
        raise NotImplementedError
