"""완료된 전체 문서 색인의 로컬 검증 표식."""

import json
from pathlib import Path

from haeindex.paths import artifact

VERSION = 1


def load(doc_id: str) -> dict | None:
    path = artifact(doc_id, "index-manifest.json")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) and value.get("version") == VERSION else None


def save(doc_id: str, *, pages: int, chunks: int) -> Path:
    path = artifact(doc_id, "index-manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"version": VERSION, "doc_id": doc_id, "pages": pages, "chunks": chunks},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path
