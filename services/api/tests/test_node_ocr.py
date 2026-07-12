"""Unit tests for the OCR vision-transcription pass (plan Task 19 / todo Task 20).

`transcribe` is the first of the three digitization passes (SPEC §4.2): it sends an
attachment image to the multimodal model under a strict "transcribe the visible text
verbatim, output text only" prompt and returns the raw transcription, with any
reasoning trace or narration removed. The node is a standalone async function here —
the LangGraph state adapter that wires it into the graph lands with the later
digitization tasks — so these tests exercise it directly against a deterministic
`FakeLLM`, never the host model (SPEC §10/§12).

The behaviours pinned here are the acceptance criteria (SPEC §4.2 / Appendix A):

- a stubbed model response carrying a `thinking` block → only the verbatim
  transcription survives (both the inline `<think>` and the CLI `Thinking… …done
  thinking.` framing the real model emits);
- a clean transcription passes through unchanged (bar surrounding whitespace);
- the image reaches the model on the multimodal `images` path;
- the prompt forces verbatim, text-only output (no description/interpretation, which
  Appendix A shows the model defaults to);
- the OCR pass overrides the project-wide reason-by-default and asks the model **not**
  to reason (`think=False`): transcription is mechanical copying, and reasoning tempts
  the model to interpret rather than transcribe.
"""

from __future__ import annotations

from app.graph.nodes.ocr import transcribe
from app.llm.fake import FakeLLM

# A stand-in base64 image payload. Its content is irrelevant to the fake — the tests
# assert only that the node forwards it on the multimodal `images` path.
_IMAGE = "aGVsbG8="

# The verbatim text the model is expected to return; the tests assert it survives the
# trace-stripping intact, whatever framing the model wraps around it.
_TRANSCRIPTION = "PAY TO THE ORDER OF John Doe $1,250.00"


async def test_inline_think_block_is_stripped() -> None:
    """An inline `<think>` reasoning block is removed, leaving only the transcription."""
    trace = "<think>The image shows a cheque. I should read the amounts.</think>"

    result = await transcribe(_IMAGE, FakeLLM(f"{trace}{_TRANSCRIPTION}"))

    assert result == _TRANSCRIPTION


async def test_cli_thinking_framing_is_stripped() -> None:
    """The CLI `Thinking… …done thinking.` framing the real model emits is removed.

    Appendix A records that `gemma4:12b` in thinking mode wraps its reasoning in this
    CLI framing; the OCR pass must keep only the final answer.
    """
    response = (
        "Thinking...\nThe document is a cheque, I will transcribe it.\n"
        f"...done thinking.\n{_TRANSCRIPTION}"
    )

    result = await transcribe(_IMAGE, FakeLLM(response))

    assert result == _TRANSCRIPTION


async def test_clean_transcription_passes_through() -> None:
    """A response with no reasoning trace comes back unchanged bar surrounding whitespace."""
    result = await transcribe(_IMAGE, FakeLLM(f"  {_TRANSCRIPTION}\n"))

    assert result == _TRANSCRIPTION


async def test_image_reaches_the_model() -> None:
    """The attachment is forwarded to the model on the multimodal `images` path."""
    llm = FakeLLM(_TRANSCRIPTION)

    await transcribe(_IMAGE, llm)

    assert llm.calls[0]["images"] == [_IMAGE]


async def test_prompt_forces_verbatim_text_only() -> None:
    """The prompt asks for a verbatim transcription, not a description of the image.

    Appendix A shows the model *describes/interprets* an image left to its own prompt,
    so the OCR prompt must explicitly demand verbatim, text-only output.
    """
    llm = FakeLLM(_TRANSCRIPTION)

    await transcribe(_IMAGE, llm)

    prompt = llm.calls[0]["prompt"].lower()
    assert "verbatim" in prompt
    assert "do not describe" in prompt


async def test_ocr_pass_does_not_request_reasoning() -> None:
    """The OCR pass overrides reason-by-default and sends `think=False`.

    Transcription is mechanical copying; reasoning only tempts the model to interpret
    the image (SPEC App. A) and produces a trace to strip. This pins the per-node
    override so a later change cannot silently flip it back on.
    """
    llm = FakeLLM(_TRANSCRIPTION)

    await transcribe(_IMAGE, llm)

    assert llm.calls[0]["think"] is False
