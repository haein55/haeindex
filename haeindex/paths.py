import re
from pathlib import Path

WORK = Path("work")


def slugify(pdf_path: Path) -> str:
    stem = pdf_path.stem.lower()
    slug = re.sub(r"[^a-z0-9가-힣]+", "-", stem).strip("-")
    if not slug:
        raise ValueError(f"doc_id 를 만들 수 없다: {pdf_path.name}")
    return slug[:60]


def work_dir(doc_id: str) -> Path:
    return WORK / doc_id


def artifact(doc_id: str, name: str) -> Path:
    return work_dir(doc_id) / name
