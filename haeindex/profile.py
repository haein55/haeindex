import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from haeindex.blocks import page_blocks
from haeindex.geometry import column_counts, stable_columns
from haeindex.headings import (
    SeqStat,
    body_ceiling,
    detect_by_font,
    heading_sequences,
    modal_size,
    repeated_texts,
    weight_coverage,
)
from haeindex.load_pdf import PDFPage, cid_ratio, doubled_ratio, load_pages, page_count
from haeindex.paths import artifact, slugify

HeadingMethod = Literal["font", "numbering", "none"]

MIN_PAGES = 3
MIN_HEADINGS = 5
DOUBLED_THRESHOLD = 0.3
NO_SPACE_THRESHOLD = 0.02
SPACE_RATIO_SYNTHESIZE = 0.4
SPACE_RATIO_BACKSTOP = 0.7
SAMPLE_PAGES = 12
HEAD_PAGES = 120


class Profile(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    source: str
    n_pages: int
    head_pages: int
    page_w: float
    page_h: float

    cid_ratio: float
    doubled_ratio: float
    space_char_ratio: float
    body_modal: float
    body_ceiling: float
    n_columns: int
    weight_coverage: float
    sequences: list[SeqStat] = Field(default_factory=list)

    heading_method: HeadingMethod
    numbering_pattern: str | None = None
    undouble: bool = False
    space_ratio: float = SPACE_RATIO_BACKSTOP
    estimated_recall: float | None = None
    why: dict[str, str] = Field(default_factory=dict)

    def save(self) -> Path:
        path = artifact(self.doc_id, "profile.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, doc_id: str) -> "Profile":
        path = artifact(doc_id, "profile.json")
        if not path.exists():
            raise FileNotFoundError(f"{path} 가 없다. `haeindex profile <pdf>` 를 먼저 돌려라")
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


def space_char_ratio(pages: Sequence[PDFPage]) -> float:
    total = sum(len(p.chars) for p in pages)
    if not total:
        return 0.0
    spaces = sum(1 for p in pages for c in p.chars if c.text.isspace())
    return spaces / total


def decide(
    *,
    n_pages: int,
    sequences: Sequence[SeqStat],
    n_font_headings: int,
) -> tuple[HeadingMethod, str | None, dict[str, str]]:
    why: dict[str, str] = {}
    if n_pages < MIN_PAGES:
        why["heading_method"] = f"{n_pages}쪽뿐이라 문서 전체 통계가 퇴화한다 → none"
        return ("none", None, why)
    if sequences:
        best = sequences[0]
        why["heading_method"] = (
            f"번호 수열 {best.pattern} 이 최장런 {best.longest_run}·연속성 "
            f"{best.contiguity:.2f} 로 자기검증된다 → numbering"
        )
        return ("numbering", best.pattern, why)
    if n_font_headings >= MIN_HEADINGS:
        why["heading_method"] = f"수열이 없고 글꼴·크기로 제목 {n_font_headings}개를 찾았다 → font"
        return ("font", None, why)
    why["heading_method"] = (
        f"수열도 없고 글꼴·크기 제목도 {n_font_headings}개뿐이다 → none (구조가 없는 문서다)"
    )
    return ("none", None, why)


def build(pdf: Path, *, head: int = HEAD_PAGES, sample: int = SAMPLE_PAGES) -> Profile:
    total = page_count(pdf)
    head_n = min(total, head)
    head_pages = load_pages(pdf, list(range(1, head_n + 1)))

    step = max(1, head_n // sample)
    sampled = head_pages[::step][:sample] or head_pages

    chars = [c for p in sampled for c in p.chars]
    doubled = doubled_ratio(chars)
    undouble = doubled >= DOUBLED_THRESHOLD

    n_columns = stable_columns(column_counts(sampled))
    spaces = space_char_ratio(sampled)

    space_ratio = SPACE_RATIO_SYNTHESIZE if spaces < NO_SPACE_THRESHOLD else SPACE_RATIO_BACKSTOP
    blocks = []
    for p in head_pages:
        blocks += page_blocks(p, space_ratio=space_ratio)

    ceiling = body_ceiling(blocks)
    drop = repeated_texts(blocks)
    seqs = heading_sequences(blocks, fix_doubled=undouble)
    n_font = len(detect_by_font(blocks, ceiling=ceiling, drop=drop))

    method, pattern, why = decide(n_pages=total, sequences=seqs, n_font_headings=n_font)
    why["columns"] = f"쪽별 단 수의 과반값 → {n_columns}단"
    why["body_ceiling"] = (
        f"문자 80% 를 덮는 크기의 상한 {ceiling:g}pt (최빈 {modal_size(blocks):g}pt)"
    )
    if space_ratio != SPACE_RATIO_BACKSTOP:
        why["space_ratio"] = (
            f"공백 문자 비율 {spaces:.2f} < {NO_SPACE_THRESHOLD} → 공백을 합성해야 한다"
            f" (space_ratio {space_ratio})"
        )
    if undouble:
        why["undouble"] = f"이중 렌더 비율 {doubled:.2f} ≥ {DOUBLED_THRESHOLD} → 숫자까지 되돌린다"

    return Profile(
        doc_id=slugify(pdf),
        source=pdf.name,
        n_pages=total,
        head_pages=head_n,
        page_w=head_pages[0].width,
        page_h=head_pages[0].height,
        cid_ratio=round(cid_ratio(chars), 4),
        doubled_ratio=round(doubled, 3),
        space_char_ratio=round(spaces, 3),
        body_modal=modal_size(blocks),
        body_ceiling=ceiling,
        n_columns=n_columns,
        weight_coverage=round(weight_coverage(blocks), 2),
        sequences=list(seqs),
        heading_method=method,
        numbering_pattern=pattern,
        undouble=undouble,
        space_ratio=space_ratio,
        estimated_recall=round(seqs[0].contiguity, 3) if seqs else None,
        why=why,
    )
