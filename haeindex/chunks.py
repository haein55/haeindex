import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from haeindex.blocks import Block
from haeindex.headings import Heading, undouble

TITLE_SEP = " > "
CONTEXT_SEP = "\n\n"
DEFAULT_MAX_CHARS = 1200
DEFAULT_MIN_CHARS = 120
_TOKEN = re.compile(r"[가-힣]|[a-zA-Z]+|\d+|[^\s\w]")


def token_len_ko(text: str) -> int:
    return len(_TOKEN.findall(text))


class Section(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    section_id: str
    seq: int
    title: str
    path: str
    depth: int
    parent_id: str | None
    start_page: int
    end_page: int
    body: str
    block_count: int


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    chunk_id: str
    seq: int
    section_ids: list[str] = Field(default_factory=list)
    title: str
    path: str
    depth: int
    page: int
    end_page: int
    body: str
    text: str
    char_len: int
    token_len: int


def _clean(text: str, fix_doubled: bool) -> str:
    return undouble(text) if fix_doubled else text


def build_sections(
    doc_id: str,
    blocks: Sequence[Block],
    headings: Sequence[Heading],
    *,
    fix_doubled: bool = False,
) -> list[Section]:
    if not headings:
        return []
    by_pos = {(h.page_index, h.order): h for h in headings}
    ordered = sorted(blocks, key=lambda b: (b.page_index, b.order))
    marks = [i for i, b in enumerate(ordered) if (b.page_index, b.order) in by_pos]

    out: list[Section] = []
    stack: list[tuple[int, str, str]] = []
    for seq, start in enumerate(marks):
        head = by_pos[(ordered[start].page_index, ordered[start].order)]
        end = marks[seq + 1] if seq + 1 < len(marks) else len(ordered)
        body_blocks = ordered[start + 1 : end]

        while stack and stack[-1][0] >= head.level:
            stack.pop()
        section_id = f"{doc_id}#s{seq:04d}"
        title = _clean(head.text, fix_doubled)
        path = TITLE_SEP.join([*(t for _, t, _ in stack), title])

        out.append(
            Section(
                doc_id=doc_id,
                section_id=section_id,
                seq=seq,
                title=title,
                path=path,
                depth=len(stack),
                parent_id=stack[-1][2] if stack else None,
                start_page=ordered[start].page_index,
                end_page=body_blocks[-1].page_index if body_blocks else ordered[start].page_index,
                body="\n".join(_clean(b.text, fix_doubled) for b in body_blocks),
                block_count=len(body_blocks),
            )
        )
        stack.append((head.level, title, section_id))
    return out


def _split_body(body: str, max_chars: int) -> list[str]:
    if len(body) <= max_chars:
        return [body]
    pieces: list[str] = []
    buf: list[str] = []
    size = 0
    for line in body.split("\n"):
        if buf and size + len(line) + 1 > max_chars:
            pieces.append("\n".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += len(line) + 1
    if buf:
        pieces.append("\n".join(buf))
    return pieces


def _make(
    doc_id: str,
    seq: int,
    *,
    section_ids: Sequence[str],
    title: str,
    path: str,
    depth: int,
    page: int,
    end_page: int,
    body: str,
    prepend_context: bool,
) -> Chunk:
    text = f"{path}{CONTEXT_SEP}{body}" if prepend_context and path else body
    return Chunk(
        doc_id=doc_id,
        chunk_id=f"{doc_id}#c{seq:04d}",
        seq=seq,
        section_ids=list(section_ids),
        title=title,
        path=path,
        depth=depth,
        page=page,
        end_page=end_page,
        body=body,
        text=text,
        char_len=len(body),
        token_len=token_len_ko(body),
    )


def chunk_sections(
    sections: Sequence[Section],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
    prepend_context: bool = True,
) -> list[Chunk]:
    out: list[Chunk] = []
    pending: list[Section] = []

    def flush(group: list[Section]) -> None:
        if not group:
            return
        head = group[0]
        body = "\n".join(s.body for s in group if s.body.strip())
        if len(group) > 1:
            body = "\n".join(
                part for s in group for part in ([s.title, s.body] if s is not head else [s.body])
            )
        for piece in _split_body(body, max_chars):
            if not piece.strip():
                continue
            out.append(
                _make(
                    head.doc_id,
                    len(out),
                    section_ids=[s.section_id for s in group],
                    title=head.title,
                    path=head.path,
                    depth=head.depth,
                    page=head.start_page,
                    end_page=group[-1].end_page,
                    body=piece,
                    prepend_context=prepend_context,
                )
            )

    for s in sections:
        pending.append(s)
        if sum(len(x.body) for x in pending) >= min_chars:
            flush(pending)
            pending = []
    flush(pending)
    return out


def chunk_fixed(
    doc_id: str,
    blocks: Sequence[Block],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    fix_doubled: bool = False,
) -> list[Chunk]:
    ordered = sorted(blocks, key=lambda b: (b.page_index, b.order))
    out: list[Chunk] = []
    buf: list[Block] = []
    size = 0

    def flush() -> None:
        nonlocal buf, size
        if not buf:
            return
        body = "\n".join(_clean(b.text, fix_doubled) for b in buf)
        if body.strip():
            out.append(
                _make(
                    doc_id,
                    len(out),
                    section_ids=[],
                    title=f"p{buf[0].page_index}",
                    path="",
                    depth=0,
                    page=buf[0].page_index,
                    end_page=buf[-1].page_index,
                    body=body,
                    prepend_context=False,
                )
            )
        buf, size = [], 0

    for b in ordered:
        text = _clean(b.text, fix_doubled)
        if buf and size + len(text) + 1 > max_chars:
            flush()
        buf.append(b)
        size += len(text) + 1
    flush()
    return out


class ChunkReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    method: str
    n_sections: int
    n_chunks: int
    median_chars: int
    max_chars: int
    oversize: int
    empty_sections: int
    uncovered_sections: list[str] = Field(default_factory=list)


def report(
    doc_id: str,
    method: str,
    sections: Sequence[Section],
    chunks: Sequence[Chunk],
    limit: int,
) -> ChunkReport:
    lens = sorted(c.char_len for c in chunks) or [0]
    covered = {sid for c in chunks for sid in c.section_ids}
    return ChunkReport(
        doc_id=doc_id,
        method=method,
        n_sections=len(sections),
        n_chunks=len(chunks),
        median_chars=lens[len(lens) // 2],
        max_chars=lens[-1],
        oversize=sum(1 for n in lens if n > limit),
        empty_sections=sum(1 for s in sections if not s.body.strip()),
        uncovered_sections=[
            s.section_id for s in sections if s.body.strip() and s.section_id not in covered
        ],
    )
