"""Fusion pass: message + attachment summary → the fused KB query (plan Task 21 / todo Task 22).

The third and final digitization pass (SPEC §4.2). The attachment is *part of the
message*, not a separate thing, so the model produces one concise **fused search query**
that combines the customer's question with a short summary of any attachment content.
This fused query — never the raw transcription — is what the agent hands to the KB
connector for retrieval (§4.4). A ticket with no attachment still runs the pass and
fuses from the message alone (project decision, todo Task 22): the model condenses even
a text-only message into a keyword query.

Two guarantees shape it:

- **Never empty.** The fused query is what searches the KB, so an empty query would
  search for nothing. Whenever a message (or a transcription) exists the result is
  non-empty: an empty or trace-only model reply falls back to the raw message (SPEC §4.2
  acceptance).
- **Trace-stripped.** Unlike the customer-facing draft — which *flags* a leaked reasoning
  trace rather than scrubbing it, leaving the rep to decide — the fused query is
  machine-consumed (fed straight to KB search), so a leaked trace would silently pollute
  retrieval. It is therefore stripped defensively (todo Task 10 util), the same reasoning
  that has the OCR pass strip its output.

`render_extracted_facts` turns the validated `ExtractionResult` the extraction pass
produced into the plain-language digest that becomes both the `extracted_facts` signal
triage and the rep see and the attachment summary fed into fusion here — one renderer,
one format.

These are plain functions, independent of LangGraph; the `ocr_extract` node that wires
the three passes together lives in the workflow assembly. Fusion keeps the project-wide
reason-by-default (`think=True`): picking the salient search terms from a message and a
document summary is interpretive. The prompt lives in-repo via the prompt registry for
now; Langfuse-managed resolution is deferred to Task 28.
"""

from __future__ import annotations

from app.llm.base import LLM
from app.llm.thinking import strip_thinking
from app.prompts.registry import get_prompt
from app.schemas.extraction import ExtractionResult


def render_extracted_facts(result: ExtractionResult) -> str:
    """Render an `ExtractionResult` as a plain-language digest for triage, the rep, and fusion.

    Lists only the structured fields that carry values (so there is never a dangling
    `Amounts:` header on a sparse extraction), notes a low-confidence flag in plain
    language, and always ends with the verbatim raw text — the one field that is never
    dropped, so the rep can always check the facts against what the document actually
    said. The same digest is fed to the fusion pass as the attachment summary.
    """
    lines: list[str] = []
    if result.doc_type:
        lines.append(f"Document type: {result.doc_type}")
    if result.amounts:
        lines.append(f"Amounts: {', '.join(result.amounts)}")
    if result.dates:
        lines.append(f"Dates: {', '.join(result.dates)}")
    if result.names:
        lines.append(f"Names: {', '.join(result.names)}")
    if result.references:
        lines.append(f"References: {', '.join(result.references)}")
    if result.low_confidence:
        lines.append("(low confidence — the model was unsure; verify against the raw text)")
    lines.append(f"Raw text:\n{result.raw_text}")
    return "\n".join(lines)


def _build_prompt(message: str, attachment_summary: str | None) -> str:
    """Fill the registered fusion template with the message and optional attachment block.

    When an attachment summary is provided it is rendered under an `Attached document
    summary:` label; with none, the section is empty so the prompt never shows a dangling
    header (mirroring triage's handling of an absent facts block).
    """
    attachment = (
        f"\nAttached document summary:\n{attachment_summary}\n" if attachment_summary else ""
    )
    return get_prompt("fuse").format(message=message, attachment=attachment)


async def fuse_query(message: str, llm: LLM, *, attachment_summary: str | None = None) -> str:
    """Produce the concise fused KB search query from the message and optional attachment summary.

    Sends the in-repo fusion prompt (with `think=True`) to `llm`, strips any leaked
    reasoning trace from the reply (the query is machine-consumed by KB search), and
    returns the trimmed query. Falls back to the raw `message` — or, failing that, the
    attachment summary — when the model returns nothing usable, so the fused query is
    **never empty whenever a message or transcription exists** (SPEC §4.2). The
    `ocr_extract` node supplies the attachment summary for a ticket with a digitized
    attachment and `None` for a text-only ticket.
    """
    prompt = _build_prompt(message, attachment_summary)
    raw = await llm.generate(prompt, think=True)
    query = strip_thinking(raw).strip()
    if query:
        return query
    # The model returned nothing usable (empty or a pure reasoning trace): fall back so
    # KB search never receives a blank query.
    return message.strip() or (attachment_summary or "").strip()
