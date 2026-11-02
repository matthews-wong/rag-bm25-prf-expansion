"""Load the JSONL corpus and query files."""

import json
from pathlib import Path


class CorpusError(ValueError):
    """Raised when a corpus or query file is malformed."""


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"{path}:{number}: invalid JSON ({exc.msg})") from exc
            if not isinstance(row, dict):
                raise CorpusError(f"{path}:{number}: expected an object, got {type(row).__name__}")
            rows.append(row)
    return rows


def load_corpus(path: Path) -> dict[str, str]:
    docs: dict[str, str] = {}
    for row in read_jsonl(path):
        doc_id, text = row.get("id"), row.get("text")
        if not isinstance(doc_id, str) or not isinstance(text, str):
            raise CorpusError(f"{path}: every row needs string 'id' and 'text', got {row!r}")
        if doc_id in docs:
            raise CorpusError(f"{path}: duplicate document id {doc_id!r}")
        docs[doc_id] = text
    return docs
