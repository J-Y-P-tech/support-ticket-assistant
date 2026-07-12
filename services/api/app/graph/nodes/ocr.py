"""OCR node: an attachment image → a verbatim text transcription (plan Task 19 / todo Task 20).

The first of the three digitization passes (SPEC §4.2). It sends the attachment image
to the single multimodal model under a strict "transcribe the visible text verbatim,
output text only" prompt and returns the raw transcription. The later passes turn that
text into a validated `ExtractionResult` (todo Task 21) and a fused search query (todo
Task 22); this pass is only the read.

Two guarantees shape it, both from SPEC §4.2 / Appendix A:

- **Verbatim, not described.** Left to its own prompt the model *interprets* an image
  rather than transcribing it, so the OCR prompt (registry `"ocr"`) forces
  character-for-character, description-free output.
- **No reasoning trace.** The pass overrides the project-wide reason-by-default and
  asks the model **not** to reason (`think=False`): transcription is mechanical copying,
  and reasoning only tempts interpretation and produces a trace to remove. As a safety
  net for a model that reasons anyway and leaks the trace into its answer, `strip_thinking`
  is applied defensively — the one place this fallback is used on purpose (todo Task 10).

This is a plain async function, independent of LangGraph; the state adapter that wraps
it into a graph node is added with the later digitization tasks. The prompt lives in-repo
via the prompt registry for now; Langfuse-managed resolution is deferred to Task 28.
"""

from __future__ import annotations

from app.llm.base import LLM
from app.llm.thinking import strip_thinking
from app.prompts.registry import get_prompt


async def transcribe(image: str, llm: LLM) -> str:
    """Transcribe the visible text in `image` verbatim, returning trace-free text.

    Sends the registered OCR prompt to `llm` with the base64-encoded `image` on the
    multimodal `images` path and `think=False` (transcription needs no reasoning —
    SPEC §4.2 / App. A). The reply is passed through `strip_thinking` defensively, so a
    model that reasons despite the flag and bakes the trace into its answer still yields
    a clean transcription. Returns the transcription text (empty when the image holds no
    readable text, per the prompt).
    """
    raw = await llm.generate(get_prompt("ocr"), images=[image], think=False)
    return strip_thinking(raw)
