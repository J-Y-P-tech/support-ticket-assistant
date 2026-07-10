"""Mock-KB provider: keyword lookup over curated canned answers (SPEC §4.4, §14.2).

Ships as the demo `KBProvider`. The curated answers live as JSON files in
`services/kb_mcp/mock_kb/`; each becomes a citable source when its indexed terms
overlap the search query. No RAG / no vectors — a real embedding provider is a
future drop-in behind the same interface (SPEC §14.1).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from providers.base import KBProvider

# Curated answer files (the "mock RAG" source data) sit alongside the service,
# one directory up from the providers package (SPEC §14.2).
_DATA_DIR = Path(__file__).resolve().parent.parent / "mock_kb"

_WORD_RE = re.compile(r"[a-z0-9]+")

# Common words that carry no retrieval signal; dropped before overlap scoring so a
# shared "how"/"my"/"the" never counts as a match — this keeps the no-confident
# path honest (a query of only these terms matches nothing).
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "is",
        "are",
        "am",
        "was",
        "were",
        "be",
        "been",
        "do",
        "does",
        "did",
        "i",
        "me",
        "my",
        "we",
        "you",
        "your",
        "it",
        "this",
        "that",
        "with",
        "can",
        "cant",
        "cannot",
        "could",
        "would",
        "should",
        "will",
        "not",
        "no",
        "how",
        "what",
        "when",
        "where",
        "why",
        "who",
        "which",
        "if",
        "so",
        "at",
        "as",
        "by",
        "from",
        "about",
        "into",
        "out",
        "up",
        "down",
        "then",
        "than",
        "they",
        "he",
        "she",
        "please",
        "help",
        "need",
        "want",
        "get",
        "got",
        "have",
        "has",
        "had",
        "just",
        "there",
        "here",
        "some",
        "any",
    }
)


def _tokenize(text: str) -> set[str]:
    """Lowercase the text to a set of significant words (stopwords removed)."""
    return {word for word in _WORD_RE.findall(text.lower()) if word not in _STOPWORDS}


def _load_articles(data_dir: Path) -> list[dict[str, Any]]:
    """Read every curated answer and precompute its match terms (title + keywords)."""
    articles: list[dict[str, Any]] = []
    for path in sorted(data_dir.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        terms = _tokenize(raw["title"] + " " + " ".join(raw.get("keywords", [])))
        articles.append({"raw": raw, "terms": terms})
    return articles


def _to_source(raw: dict[str, Any]) -> dict[str, Any]:
    """Project a curated answer into a `KBSource`-shaped dict."""
    return {
        "id": raw["id"],
        "title": raw["title"],
        "text": raw["text"],
    }


class MockKBProvider(KBProvider):
    """Keyword-overlap lookup over the curated `mock_kb/` answers."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._articles = _load_articles(data_dir if data_dir is not None else _DATA_DIR)

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        query_terms = _tokenize(query)
        if not query_terms:
            return []
        matches = [
            (len(query_terms & article["terms"]), article["raw"])
            for article in self._articles
            if query_terms & article["terms"]
        ]
        # Best overlap first; ties broken by id so the ranking is deterministic.
        matches.sort(key=lambda match: (-match[0], match[1]["id"]))
        return [_to_source(raw) for _, raw in matches[:limit]]
