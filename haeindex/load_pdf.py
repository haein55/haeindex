import re
import statistics
from collections import Counter
from pathlib import Path

import pdfplumber
from pydantic import BaseModel, ConfigDict, Field

CID_TOKEN = re.compile(r"\(cid:\d+\)")
REPLACEMENT = "�"


class PDFChar(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    size: float
    fontname: str


class PDFPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    page_index: int
    width: float
    height: float
    chars: list[PDFChar] = Field(default_factory=list)
    rotated: list[PDFChar] = Field(default_factory=list)


def load_pages(pdf_path: Path, pages: list[int] | None = None) -> list[PDFPage]:
    out: list[PDFPage] = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        wanted = pages or range(1, total + 1)
        bad = [n for n in wanted if not 1 <= n <= total]
        if bad:
            raise ValueError(f"{pdf_path.name} 은 {total}쪽인데 {bad} 를 요청했다")
        for n in wanted:
            page = pdf.pages[n - 1]
            upright, rot = [], []
            for c in page.chars:
                target = upright if c.get("upright", True) else rot
                target.append(
                    PDFChar(
                        text=str(c["text"]),
                        x0=round(float(c["x0"]), 2),
                        x1=round(float(c["x1"]), 2),
                        top=round(float(c["top"]), 2),
                        bottom=round(float(c["bottom"]), 2),
                        size=round(float(c["size"]), 2),
                        fontname=str(c["fontname"]),
                    )
                )
            out.append(
                PDFPage(
                    page_index=n,
                    width=round(float(page.width), 2),
                    height=round(float(page.height), 2),
                    chars=upright,
                    rotated=rot,
                )
            )
            page.flush_cache()
    return out


def page_count(pdf_path: Path) -> int:
    with pdfplumber.open(pdf_path) as pdf:
        return len(pdf.pages)


def cid_ratio(chars: list[PDFChar]) -> float:
    if not chars:
        return 1.0
    total = sum(len(c.text) for c in chars)
    if total == 0:
        return 1.0
    bad = sum(
        sum(len(m.group()) for m in CID_TOKEN.finditer(c.text)) + c.text.count(REPLACEMENT)
        for c in chars
    )
    return bad / total


def line_bin(chars: list[PDFChar], ratio: float = 0.6) -> float:
    if not chars:
        return 1.0
    h = statistics.median(c.bottom - c.top for c in chars)
    return max(0.5, h * ratio)


def reading_order(chars: list[PDFChar]) -> list[PDFChar]:
    bin_h = line_bin(chars)
    return sorted(chars, key=lambda c: (round((c.top + c.bottom) / 2 / bin_h), c.x0))


def doubled_ratio(chars: list[PDFChar]) -> float:
    content = [c for c in chars if not c.text.isspace()]
    if len(content) < 20:
        return 0.0
    t = "".join(c.text for c in reading_order(content))
    dup = sum(
        1 for a, b in zip(t, t[1:], strict=False) if a == b and not a.isdigit() and a.isalpha()
    )
    return dup / (len(t) - 1)


def top_fonts(pages: list[PDFPage], n: int = 6) -> dict[str, int]:
    pairs: Counter[str] = Counter()
    for p in pages:
        for c in p.chars:
            pairs[f"{c.fontname.split('+')[-1]}@{c.size:g}"] += 1
    return dict(pairs.most_common(n))
