import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Annotated

import typer

from haeindex.agent import match_doc, summarize_traces
from haeindex.agent import run as run_agent
from haeindex.answer import MESSAGES, should_refuse
from haeindex.answer import answer as run_answer
from haeindex.blocks import build_blocks, crosses_gutter, page_blocks, size_tiers
from haeindex.chunks import (
    build_sections,
    chunk_fixed,
    chunk_sections,
    report,
)
from haeindex.diagnose import FIXES, diagnose, tally
from haeindex.document_cards import (
    DOC_CANDIDATES,
    doc_card_count,
    ensure_doc_index,
    generate_card,
    index_card,
    search_cards,
    search_lexical_docs,
)
from haeindex.enrich import enrich_many
from haeindex.evaluate import Goldset, paired_bootstrap, score_query, summarize
from haeindex.evidence import select_evidence
from haeindex.features import gen_many
from haeindex.geometry import column_counts, in_band, page_gutters, stable_columns
from haeindex.headings import (
    assign_levels,
    body_ceiling,
    detect_by_font,
    detect_by_numbering,
    heading_sequences,
    modal_size,
    repeated_texts,
    weight_coverage,
)
from haeindex.index import client, doc_counts, ensure_index, index_chunks, replace_doc
from haeindex.index_manifest import save as save_index_manifest
from haeindex.listwise import DEPTH as LISTWISE_DEPTH
from haeindex.listwise import rerank as listwise_rerank
from haeindex.load_pdf import (
    cid_ratio,
    doubled_ratio,
    line_bin,
    load_pages,
    page_count,
    reading_order,
    top_fonts,
)
from haeindex.models import model_client
from haeindex.paths import slugify
from haeindex.profile import Profile, build
from haeindex.rerank import Model as RerankModel
from haeindex.rerank import fit, harvest, mrr_of
from haeindex.rerank import rerank as apply_rerank
from haeindex.routing import route_question
from haeindex.search import CANDIDATE_K
from haeindex.search import search as run_search

app = typer.Typer(add_completion=False, no_args_is_help=True, help="PDF → 검색 → 답변")


def _resolve_docs(os_client, patterns: list[str]) -> list[str]:
    if not patterns:
        return []
    known = sorted(doc_counts(os_client))
    out: list[str] = []
    for pat in patterns:
        hit = match_doc(pat, known)
        if not hit:
            raise typer.BadParameter(
                f"--doc {pat!r} 가 어느 문서와도 안 맞는다.\n색인된 문서: {known}"
            )
        out += [d for d in hit if d not in out]
    return out


def _sample(n_pages: int, k: int = 8) -> list[int]:
    if n_pages <= k:
        return list(range(1, n_pages + 1))
    return sorted({max(1, round(n_pages * i / (k + 1))) for i in range(1, k + 1)})


def _fmt(counts: Counter[int]) -> str:
    return " ".join(f"{n}단×{c}" for n, c in sorted(counts.items()))


@app.command()
def pages(
    pdf: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    sample: Annotated[int, typer.Option("--sample")] = 8,
    chars: Annotated[int, typer.Option("--chars", help="글자를 하나씩 몇 개 찍을까")] = 0,
    page: Annotated[int, typer.Option("--page", help="--chars 로 볼 쪽 번호")] = 0,
) -> None:
    n = page_count(pdf)
    if chars:
        target = page or 1
        if not 1 <= target <= n:
            raise typer.BadParameter(f"{target}쪽은 없다 (1~{n})")
        one = load_pages(pdf, [target])[0]
        typer.echo(f"{pdf.name} p{target}  문자 {len(one.chars):,}개 · 읽는 순서 앞 {chars}개")
        typer.echo(f"  {'글자':<5}{'x0':>8}{'x1':>8}{'top':>8}{'bottom':>9}{'pt':>7}  글꼴")
        for c in reading_order(one.chars)[:chars]:
            shown = "·" if c.text.isspace() else c.text
            typer.echo(
                f"  {shown:<5}{c.x0:>8.1f}{c.x1:>8.1f}{c.top:>8.1f}"
                f"{c.bottom:>9.1f}{c.size:>7.1f}  {c.fontname}"
            )
        typer.echo(f"\n  줄 묶는 단위 높이 line_bin = {line_bin(one.chars):.2f}pt")
        return

    docs_ = load_pages(pdf, _sample(n, sample) if sample else None)
    chars = [c for p in docs_ for c in p.chars]
    rot = sum(len(p.rotated) for p in docs_)

    typer.echo(f"{pdf.name}")
    typer.echo(f"  doc_id      {slugify(pdf)}")
    typer.echo(f"  쪽수         {n}  (표본 {len(docs_)}쪽)")
    typer.echo(f"  문자          {len(chars):,}  (회전 {rot})")
    typer.echo(f"  페이지 크기    {docs_[0].width:g} × {docs_[0].height:g} pt")
    typer.echo(f"  cid 비율      {cid_ratio(chars):.4f}")
    typer.echo(f"  이중글자 비율  {doubled_ratio(chars):.2f}")
    for name, cnt in top_fonts(docs_).items():
        typer.echo(f"    {cnt:>7,}  {name}")


@app.command()
def columns(
    pdf: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    sample: Annotated[int, typer.Option("--sample")] = 8,
    header: Annotated[float, typer.Option("--header")] = 0.0,
    footer: Annotated[float, typer.Option("--footer")] = 1.0,
    bins: Annotated[int, typer.Option("--bins")] = 160,
    page: Annotated[int, typer.Option("--page", help="이 쪽의 히스토그램을 그린다")] = 0,
) -> None:
    n = page_count(pdf)
    docs_ = load_pages(pdf, _sample(n, sample))
    counts = column_counts(docs_, header=header, footer=footer, bins=bins)
    typer.echo(f"{pdf.name}  (표본 {len(docs_)}쪽, bins={bins}, 밴드 {header}~{footer})")
    typer.echo(f"  페이지별      {_fmt(counts)}")
    typer.echo(f"  안정된 단 수   {stable_columns(counts)}")
    if page:
        _show_occupancy(pdf, page, bins=bins, header=header, footer=footer)


def _show_occupancy(pdf: Path, page: int, *, bins: int, header: float, footer: float) -> None:
    one = load_pages(pdf, [page])[0]
    body = [c for c in one.chars if in_band(c, one.height, header, footer)]
    gutters = page_gutters(
        one.chars, one.width, one.height, header=header, footer=footer, bins=bins
    )
    bin_w = one.width / bins
    hits = [0] * bins
    for c in body:
        for i in range(max(0, int(c.x0 / bin_w)), min(bins - 1, int(c.x1 / bin_w)) + 1):
            hits[i] += 1
    peak = max(hits) or 1

    typer.echo(
        f"\n  p{page} x 점유 히스토그램  (문자 {len(body):,} · 칸 {bin_w:.1f}pt · 최대 {peak})"
    )
    step = max(1, bins // 78)
    bars = " ▁▂▃▄▅▆▇█"
    line = "".join(
        bars[min(8, int(hits[i] / peak * 8) + (1 if hits[i] else 0))] for i in range(0, bins, step)
    )
    typer.echo(f"    {line}")
    typer.echo(f"    0pt{' ' * max(0, len(line) - 9)}{one.width:.0f}pt")
    if gutters:
        for g0, g1 in gutters:
            typer.echo(f"    여백띠  {g0:.1f} ~ {g1:.1f}pt  (폭 {g1 - g0:.1f}pt)")
    else:
        typer.echo("    여백띠 없음 → 1단")


@app.command()
def blocks(
    pdf: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    page: Annotated[int, typer.Option("--page")] = 1,
    limit: Annotated[int, typer.Option("--limit")] = 20,
) -> None:
    p = load_pages(pdf, [page])[0]
    g = page_gutters(p.chars, p.width, p.height)
    bs = build_blocks(p, g)
    typer.echo(f"{pdf.name} p{page}  gutter {len(g)}개 → {len(g) + 1}단  블록 {len(bs)}개")
    for b in bs[:limit]:
        typer.echo(f"  [{b.column}] {b.size:>5.2f}pt {b.fontname[:18]:20} {b.text[:52]}")


def _head_blocks(pdf: Path, n_pages: int) -> list:
    total = page_count(pdf)
    pages = list(range(1, min(total, n_pages) + 1))
    out = []
    for p in load_pages(pdf, pages):
        out += page_blocks(p)
    return out


@app.command()
def headings(
    pdf: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    head: Annotated[int, typer.Option("--head", help="앞에서 연속 N쪽")] = 30,
    size_ratio: Annotated[float, typer.Option("--size-ratio")] = 1.05,
    limit: Annotated[int, typer.Option("--limit")] = 12,
) -> None:
    bs = _head_blocks(pdf, head)
    if not bs:
        typer.echo("블록이 없다")
        return
    modal = modal_size(bs)
    ceiling = body_ceiling(bs)
    drop = repeated_texts(bs)
    cov = weight_coverage(bs)

    typer.echo(f"{pdf.name}  (앞 {min(page_count(pdf), head)}쪽 · 블록 {len(bs)}개)")
    typer.echo(f"  본문 크기      최빈 {modal:g}pt · 상한 {ceiling:g}pt (문자 80%)")
    typer.echo(f"  굵기 읽힌 비율  {cov:.2f}")
    typer.echo(f"  반복 머리말     {len(drop)}개 제외")

    plain = heading_sequences(bs)
    fixed_seqs = heading_sequences(bs, fix_doubled=True)
    fixed = sum(s.observed for s in fixed_seqs) > sum(s.observed for s in plain)
    seqs = fixed_seqs if fixed else plain
    if seqs:
        for s in seqs:
            typer.echo(
                f"  수열 {s.pattern:10} 관측 {s.observed:>3} · 최대 {s.max_ordinal:>3}"
                f" · {s.pages}쪽 · 연속성 {s.contiguity:.2f}"
                f" · 최장런 {s.longest_run} · 빠짐 {s.missing[:5]}"
            )
    else:
        typer.echo("  수열          없음")

    by_font = assign_levels(detect_by_font(bs, ceiling=ceiling, size_ratio=size_ratio, drop=drop))
    typer.echo(
        f"  font 방법      제목 {len(by_font)}개  {dict(Counter(h.signal for h in by_font))}"
    )
    if seqs:
        by_num = detect_by_numbering(bs, seqs[0].pattern, fix_doubled=fixed, drop=drop)
        typer.echo(f"  numbering      제목 {len(by_num)}개  (undouble={fixed})")
        for h in by_num[:limit]:
            typer.echo(f"     p{h.page_index:>3} {h.size:>5.2f}pt {h.text[:50]}")
    else:
        for h in by_font[:limit]:
            typer.echo(
                f"     p{h.page_index:>3} L{h.level} {h.size:>5.2f}pt {h.signal:5} {h.text[:44]}"
            )


@app.command()
def profile(
    pdf: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    head: Annotated[int, typer.Option("--head")] = 120,
) -> None:
    p = build(pdf, head=head)
    path = p.save()

    typer.echo(f"{p.source}  →  {path}")
    typer.echo(f"  doc_id          {p.doc_id}")
    typer.echo(f"  쪽수             {p.n_pages}  (앞 {p.head_pages}쪽을 읽었다)")
    typer.echo(f"  페이지            {p.page_w:g} × {p.page_h:g} pt")
    typer.echo("  ── 측정 ──")
    typer.echo(f"  cid 비율         {p.cid_ratio:.4f}")
    typer.echo(f"  이중 렌더        {p.doubled_ratio:.2f}")
    typer.echo(f"  공백 문자 비율    {p.space_char_ratio:.2f}  → space_ratio {p.space_ratio}")
    typer.echo(f"  본문 크기        최빈 {p.body_modal:g}pt · 상한 {p.body_ceiling:g}pt")
    typer.echo(f"  단 수            {p.n_columns}")
    typer.echo(f"  굵기 읽힌 비율    {p.weight_coverage:.2f}")
    for s in p.sequences:
        typer.echo(
            f"  수열 {s.pattern:8} 관측 {s.observed} · 최장런 {s.longest_run}"
            f" · 연속성 {s.contiguity:.2f} · 빠짐 {s.missing[:5]}"
        )
    typer.echo("  ── 판정 ──")
    typer.echo(
        f"  제목 탐지 방법    {p.heading_method}"
        + (f" ({p.numbering_pattern})" if p.numbering_pattern else "")
    )
    typer.echo(f"  숫자 되돌리기     {p.undouble}")
    if p.estimated_recall is not None:
        typer.echo(f"  추정 recall      {p.estimated_recall:.2f}  (수열 연속성 · 라벨 0개)")
    for k, v in p.why.items():
        typer.echo(f"    {k:14} {v}")


def _chunks_for(pdf: Path, prof: Profile, max_chars: int) -> tuple[list, list, str]:
    # head_pages는 구조 판정을 위한 표본 범위일 뿐이다. 색인은 반드시 문서 전체를 읽는다.
    blocks = []
    for p in load_pages(pdf, list(range(1, prof.n_pages + 1))):
        blocks += page_blocks(p, space_ratio=prof.space_ratio)

    if prof.heading_method == "none":
        return (
            [],
            chunk_fixed(prof.doc_id, blocks, max_chars=max_chars, fix_doubled=prof.undouble),
            "fixed",
        )

    drop = repeated_texts(blocks)
    if prof.heading_method == "numbering" and prof.numbering_pattern:
        hs = detect_by_numbering(
            blocks, prof.numbering_pattern, fix_doubled=prof.undouble, drop=drop
        )
    else:
        hs = detect_by_font(blocks, ceiling=prof.body_ceiling, drop=drop)
    hs = assign_levels(hs)
    secs = build_sections(prof.doc_id, blocks, hs, fix_doubled=prof.undouble)
    return (secs, chunk_sections(secs, max_chars=max_chars), prof.heading_method)


@app.command()
def chunk(
    pdf: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    max_chars: Annotated[int, typer.Option("--max-chars")] = 1200,
    limit: Annotated[int, typer.Option("--limit")] = 4,
) -> None:
    prof = Profile.load(slugify(pdf))
    secs, chs, how = _chunks_for(pdf, prof, max_chars)
    rep = report(prof.doc_id, how, secs, chs, max_chars)

    typer.echo(f"{prof.source}  방법 {how}" + (" (폴백)" if how == "fixed" else ""))
    typer.echo(f"  절 {rep.n_sections}개 · 청크 {rep.n_chunks}개")
    typer.echo(
        f"  길이  중앙 {rep.median_chars} · 최대 {rep.max_chars} · 초과({max_chars}) {rep.oversize}"
    )
    if rep.n_sections:
        typer.echo(
            f"  본문 없는 절 {rep.empty_sections}개 · 안 덮인 절 {len(rep.uncovered_sections)}개"
        )
    for c in chs[:limit]:
        label = (c.path or c.title)[:36]
        cid = c.chunk_id.split("#")[-1]
        typer.echo(f"    [{cid}] p{c.page:>3} {c.char_len:>5}자 절{len(c.section_ids)}  {label}")
        typer.echo(f"        {c.body[:74]!r}")


@app.command()
def index(
    pdf: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    max_chars: Annotated[int, typer.Option("--max-chars")] = 1200,
    accuracy: Annotated[
        bool, typer.Option("--accuracy/--no-accuracy", help="검색 문맥·조건·절 카드 보강")
    ] = False,
    with_queries: Annotated[
        bool, typer.Option("--with-queries", help="가상 질문 생성. 청크당 LLM 1회")
    ] = False,
    with_metadata: Annotated[
        bool, typer.Option("--with-metadata", help="LLM 검색 메타데이터 생성. 청크당 1회")
    ] = False,
) -> None:
    prof = Profile.load(slugify(pdf))
    _, chs, how = _chunks_for(pdf, prof, max_chars)
    if not chs:
        raise typer.BadParameter("청크가 0개다")

    os_client = client()
    ensure_index(os_client)

    gen: list[list[str]] | None = None
    metadata = None
    card = None
    card_vector = None
    failed = 0
    metadata_failed = 0
    with model_client("enrich") as ol:
        vectors = ol.embed_batched([c.text for c in chs])
        if with_queries:
            with typer.progressbar(length=len(chs), label="가상 질문") as bar:
                gen, failed = gen_many(ol, [c.body for c in chs], lambda i, n: bar.update(1))
        if with_metadata:
            with typer.progressbar(length=len(chs), label="검색 메타데이터") as bar:
                metadata, metadata_failed = enrich_many(
                    ol, [c.body for c in chs], lambda i, n: bar.update(1)
                )
            card = generate_card(ol, prof.doc_id, prof.source, chs)
            if card is not None:
                card_vector = ol.embed([card.search_text()])[0]

    gone = replace_doc(os_client, prof.doc_id)
    ok, errors = index_chunks(os_client, chs, vectors, queries=gen, enrichments=metadata)
    if not errors:
        save_index_manifest(prof.doc_id, pages=prof.n_pages, chunks=ok)
    if with_metadata and card is not None and card_vector is not None:
        ensure_doc_index(os_client)
        index_card(os_client, card, card_vector)

    typer.echo(f"{prof.source}  방법 {how}")
    typer.echo(f"  청크 {len(chs)}개 · 임베딩 {len(vectors)}개 (dim {len(vectors[0])})")
    typer.echo(
        f"  기존 청크 {gone}건 교체 → 색인 {ok}건" + (f" · 오류 {len(errors)}" if errors else "")
    )
    if with_queries:
        typer.echo(f"  가상 질문 생성 실패 {failed}/{len(chs)} (빈 값으로 기록했다)")
    if with_metadata:
        typer.echo(
            f"  검색 메타데이터 생성 실패 {metadata_failed}/{len(chs)} (원문 색인은 유지했다)"
        )
        if card is None:
            typer.echo("  문서 카드 생성 실패 (청크 색인은 유지했다)")
        else:
            typer.echo(
                f"  문서 카드 1건 색인 · 유형 {card.document_type} · "
                f"주제 {len(card.topics)}개 · 전체 {doc_card_count(os_client)}건"
            )
    if accuracy and not errors:
        from haeindex.augmentation import enhance_document

        with model_client("enrich", num_ctx=16384) as enrich_llm:
            result = enhance_document(
                os_client, enrich_llm, enrich_llm, prof.doc_id, progress=typer.echo
            )
        typer.echo(
            f"  검색 보강: {result['accepted']}/{result['total']} 청크 · "
            f"실패 {len(result['failures'])}건"
        )
    typer.echo(f"  인덱스 전체: {doc_counts(os_client)}")


@app.command()
def card(
    pdf: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    max_chars: Annotated[int, typer.Option("--max-chars")] = 1200,
) -> None:
    """청크를 교체하지 않고 문서 카드만 생성하거나 갱신한다."""
    prof = Profile.load(slugify(pdf))
    _, chs, _ = _chunks_for(pdf, prof, max_chars)
    if not chs:
        raise typer.BadParameter("청크가 0개다")

    os_client = client()
    ensure_doc_index(os_client)
    with model_client("enrich") as ol:
        item = generate_card(ol, prof.doc_id, prof.source, chs)
        if item is None:
            raise typer.BadParameter("두 번 시도했지만 문서 카드 JSON 생성에 실패했다")
        vector = ol.embed([item.search_text()])[0]
    index_card(os_client, item, vector)
    typer.echo(
        f"{item.doc_id}  카드 색인 완료 · 유형 {item.document_type} · "
        f"주제 {len(item.topics)}개 · 전체 {doc_card_count(os_client)}건"
    )


@app.command()
def search(
    query: Annotated[str, typer.Argument()],
    doc: Annotated[list[str] | None, typer.Option("--doc", help="반복 가능")] = None,
    top_k: Annotated[int, typer.Option("--top-k")] = 5,
    explain: Annotated[bool, typer.Option("--explain/--no-explain")] = True,
    no_embed: Annotated[bool, typer.Option("--no-embed")] = False,
    listwise: Annotated[bool, typer.Option("--listwise", help="LLM 으로 후보를 재정렬")] = False,
    depth: Annotated[int, typer.Option("--depth", help="재정렬할 후보 수")] = LISTWISE_DEPTH,
) -> None:
    os_client = client()
    wanted = _resolve_docs(os_client, doc or [])
    ol = None if no_embed else model_client("analysis")
    ranked = None
    try:
        wide = CANDIDATE_K if listwise else top_k
        res = run_search(os_client, query, embedder=ol, doc_ids=wanted, top_k=wide)
        if listwise:
            if ol is None:
                raise typer.BadParameter("--listwise 는 LLM 이 필요하다. --no-embed 와 못 쓴다")
            ranked = listwise_rerank(ol, query, res.hits, depth=depth)
            res = res.model_copy(update={"hits": ranked.hits[:top_k]})
        else:
            res = res.model_copy(update={"hits": res.hits[:top_k]})
    finally:
        if ol:
            ol.close()

    typer.echo(f"query={query!r}  후보 {res.candidates}  dedupe -{res.dropped}")
    if ranked:
        typer.echo(
            f"  재정렬 depth {depth} · LLM {ranked.calls}회 · 자리바뀜 {ranked.moved}"
            + (f" · 파싱실패 {ranked.parse_fails}" if ranked.parse_fails else "")
        )
    if res.degraded:
        typer.echo(f"  강등된 leg: {res.degraded}")
    for i, h in enumerate(res.hits, 1):
        s = h.source
        legs = " ".join(f"{k}={v.rank}/{v.score:.3f}" for k, v in sorted(h.legs.items()))
        typer.echo(f"  {i}. [{s['doc_id'][:16]}] p{s['page']:>3} {legs}")
        typer.echo(f"     {(s.get('path') or s.get('title') or '')[:64]}")
        if explain:
            typer.echo(f"     {s.get('body', '')[:100]!r}")


def _show_top(res, limit: int) -> None:
    for i, h in enumerate(res.hits[:limit], 1):
        s = h.source
        legs = " ".join(f"{k}={v.rank}" for k, v in sorted(h.legs.items()))
        label = (s.get("path") or s.get("title") or "")[:56]
        typer.echo(f"    {i}. [{s['doc_id'][:14]:14}] p{s['page']:>3}  {legs:16} {label}")


def _show_diag(d, res, limit: int) -> None:
    typer.echo(f"{d.id}  [{d.bucket}]  {d.query!r}")
    typer.echo(f"  정답      {d.doc_id} · {' '.join(d.targets)}")
    typer.echo(f"  판정      {d.cause}  — {FIXES[d.cause]}")
    typer.echo(
        f"\n  정답을 덮는 청크 {len(d.covering)}개  (후보 {d.candidate_k} · top_k {d.top_k})"
    )
    for c in d.covering:
        missing = "dedupe" if c.deduped else "후보 밖"
        where = f"융합 {c.fused_rank:>3}" if c.fused_rank else missing
        legs = " ".join(f"{k} {v}" for k, v in sorted(c.legs.items())) or "-"
        typer.echo(
            f"    {c.chunk_id.split('#')[-1]:7} p{c.page:>3}-{c.end_page:<3} "
            f"{where:9} {legs:14} {c.label[:44]}"
        )
    typer.echo("\n  실제로 온 top-k")
    _show_top(res, limit)


@app.command()
def why(
    query: Annotated[str | None, typer.Argument(help="자유 질의. --id 를 쓰면 생략한다")] = None,
    id_: Annotated[str | None, typer.Option("--id", help="골든셋 문항 id")] = None,
    every: Annotated[bool, typer.Option("--all", help="골든셋 전체를 원인별로 집계")] = False,
    goldset: Annotated[Path, typer.Option("--goldset", exists=True)] = Path("goldset/all.yaml"),
    doc: Annotated[list[str] | None, typer.Option("--doc")] = None,
    top_k: Annotated[int, typer.Option("--top-k")] = 5,
    candidate_k: Annotated[int, typer.Option("--candidate-k")] = CANDIDATE_K,
    limit: Annotated[int, typer.Option("--limit")] = 8,
    baseline: Annotated[bool, typer.Option("--baseline", help="기존 검색만 진단")] = False,
    trace_file: Annotated[Path | None, typer.Option("--trace", help="전체 실행 기록 JSON")] = None,
) -> None:
    """어떤 청크가 왔는지 + 정답 청크는 어디서 사라졌는지."""
    os_client = client()
    if not (query or id_ or every):
        raise typer.BadParameter("질의를 주거나 --id 또는 --all 을 쓴다")

    if not baseline:
        known = sorted(doc_counts(os_client))
        explicit = _resolve_docs(os_client, doc or [])
        if query and not (id_ or every):
            trace = _enhanced(os_client, query, known, explicit, top_k=max(top_k, 8))
            _show_trace(trace, trace_file)
            _show_top(trace.result, limit)
            return
        gold = Goldset.load(goldset)
        picked = [q for q in gold.ranked if id_ is None or q.id == id_]
        if not picked:
            raise typer.BadParameter(f"{id_} 를 골든셋에서 못 찾았다")
        traces = []
        for q in picked:
            trace = _enhanced(os_client, q.query, known, explicit, top_k=max(top_k, 8))
            traces.append({"id": q.id, **trace.model_dump(mode="json")})
            typer.echo(f"\n{q.id}: {q.query}")
            _show_trace(trace)
            _show_top(trace.result, limit)
            d = diagnose(
                os_client,
                q,
                trace.result.hits,
                dropped_ids=trace.result.dropped_ids,
                top_k=top_k,
                candidate_k=candidate_k,
            )
            _show_diag(d, trace.result, limit)
        if trace_file:
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            trace_file.write_text(json.dumps(traces, ensure_ascii=False, indent=2))
        return
    with model_client("analysis") as ol:
        if query and not (id_ or every):
            wanted = _resolve_docs(os_client, doc or [])
            res = run_search(
                os_client, query, embedder=ol, doc_ids=wanted, top_k=limit, candidate_k=candidate_k
            )
            typer.echo(f"query={query!r}  후보 {res.candidates}  dedupe -{res.dropped}")
            _show_top(res, limit)
            return

        gold = Goldset.load(goldset)
        gold.verify(list(doc_counts(os_client)))
        picked = [q for q in gold.ranked if id_ is None or q.id == id_]
        if not picked:
            raise typer.BadParameter(f"{id_} 를 골든셋에서 못 찾았다")

        diags = []
        for q in picked:
            res = run_search(
                os_client, q.query, embedder=ol, top_k=candidate_k, candidate_k=candidate_k
            )
            d = diagnose(
                os_client,
                q,
                res.hits,
                dropped_ids=res.dropped_ids,
                top_k=top_k,
                candidate_k=candidate_k,
            )
            diags.append(d)
            if not every:
                _show_diag(d, res, limit)

    if not every:
        return

    counts = tally(diags)
    typer.echo(f"골든셋 {goldset}  ·  순위 {len(diags)}문항  ·  후보 {candidate_k} · top_k {top_k}")
    typer.echo(f"\n  {'원인':8}{'문항':>6}   고칠 곳")
    typer.echo("  " + "─" * 78)
    for cause, ids in counts.items():
        typer.echo(f"  {cause:8}{len(ids):>6}   {FIXES[cause]}")
    for cause, ids in counts.items():
        if cause != "성공" and ids:
            typer.echo(f"\n  {cause}: {', '.join(ids)}")

    typer.echo("\n  버킷 × 원인")
    buckets = sorted({d.bucket for d in diags})
    typer.echo("  " + f"{'원인':8}" + "".join(f"{b:>7}" for b in buckets))
    for cause in counts:
        row = [sum(1 for d in diags if d.bucket == b and d.cause == cause) for b in buckets]
        typer.echo(f"  {cause:8}" + "".join(f"{n:>7}" for n in row))


@app.command()
def ask(
    question: Annotated[str, typer.Argument()],
    model: Annotated[
        str | None, typer.Option("--model", help="답변 모델 이름 또는 Bedrock 모델 ID")
    ] = None,
    doc: Annotated[list[str] | None, typer.Option("--doc")] = None,
    top_k: Annotated[int, typer.Option("--top-k")] = 5,
    min_cos: Annotated[float, typer.Option("--min-cos")] = 0.78,
    num_ctx: Annotated[int, typer.Option("--num-ctx")] = 8192,
    strict: Annotated[bool, typer.Option("--strict/--no-strict")] = True,
    rerank: Annotated[bool, typer.Option("--rerank/--no-rerank")] = True,
    depth: Annotated[int, typer.Option("--depth", help="LLM 재정렬 후보 수")] = LISTWISE_DEPTH,
    auto_doc: Annotated[
        bool, typer.Option("--auto-doc/--no-auto-doc", help="질문의 파일명을 자동 인식")
    ] = True,
    doc_k: Annotated[int, typer.Option("--doc-k", help="문서 카드 후보 수")] = DOC_CANDIDATES,
    baseline: Annotated[bool, typer.Option("--baseline", help="기존 단일 패스 답변")] = False,
    trace_file: Annotated[Path | None, typer.Option("--trace", help="전체 실행 기록 JSON")] = None,
) -> None:
    os_client = client()
    known = sorted(doc_counts(os_client))
    explicit = _resolve_docs(os_client, doc or [])
    if not baseline:
        trace = _enhanced(
            os_client,
            question,
            known,
            explicit,
            model=model,
            top_k=max(top_k, 8),
            num_ctx=max(num_ctx, 16384),
        )
        _show_trace(trace, trace_file)
        return
    route = route_question(
        question,
        known,
        explicit_doc_ids=explicit,
    )
    if not auto_doc and not explicit:
        route = route.model_copy(update={"doc_ids": [], "reason": "자동 라우팅 꺼짐"})
    if route.clarification:
        typer.echo(route.clarification)
        return
    wanted = route.doc_ids
    with model_client("answer", model=model, num_ctx=num_ctx) as ol:
        if auto_doc and not wanted:
            card_hits = search_cards(os_client, question, embedder=ol, top_k=doc_k)
            lexical_docs = search_lexical_docs(os_client, question)
            card_docs = [h.doc_id for h in card_hits]
            wanted = list(dict.fromkeys([*lexical_docs, *card_docs]))
            if wanted:
                sources = []
                if lexical_docs:
                    sources.append(f"청크 BM25 top-{len(lexical_docs)}")
                if card_docs:
                    sources.append(f"문서 카드 top-{len(card_docs)}")
                route = route.model_copy(update={"doc_ids": wanted, "reason": " + ".join(sources)})
        wide = max(top_k, depth) if rerank else top_k
        res = run_search(os_client, question, embedder=ol, doc_ids=wanted, top_k=wide)
        ranked = listwise_rerank(ol, question, res.hits, depth=depth) if rerank else None
        if ranked:
            res = res.model_copy(update={"hits": ranked.hits})
        res = select_evidence(
            res,
            top_k=top_k,
            allow_multiple_docs=route.allow_multiple_docs,
        )
        ans = run_answer(ol, res, question, min_cos=min_cos, strict=strict)

    if ans.refusal is not None:
        typer.echo(MESSAGES[ans.refusal])
        typer.echo(f"  [코사인 top1 {res.top_score('knn'):.3f} · 임계 {min_cos}]")
        typer.echo(
            f"  [라우팅 {route.reason}: {', '.join(wanted) or '전체'} · "
            f"문서혼합 {'허용' if route.allow_multiple_docs else '차단'}]"
        )
        return

    typer.echo(ans.text)
    typer.echo("\n출처")
    for b in ans.context.blocks:
        if b.n in ans.cited:
            typer.echo(f"  [{b.n}] {b.doc_id} · p.{b.page} · {b.label[:52]}")
    typer.echo(
        f"\n[코사인 {res.top_score('knn'):.3f} · 컨텍스트 {len(ans.context.blocks)}블록"
        f" ~{ans.context.est_tokens}토큰]"
    )
    typer.echo(
        f"[모델 {ol.chat_model} · 라우팅 {route.reason}: {', '.join(wanted) or '전체'} · "
        f"문서혼합 {'허용' if route.allow_multiple_docs else '차단'}"
        + (f" · 언어재생성 {ans.language_retries}회" if ans.language_retries else "")
        + (f" · 인용재생성 {ans.citation_retries}회" if ans.citation_retries else "")
        + (" · 일부 문장 인용 불완전" if not ans.citations_complete else "")
        + "]"
    )


def _enhanced(os_client, question, known, explicit, *, model=None, top_k=8, num_ctx=16384):
    from haeindex.pipeline import run_pipeline

    with ExitStack() as stack:
        answer_llm = stack.enter_context(model_client("answer", model=model, num_ctx=num_ctx))
        analysis = stack.enter_context(model_client("analysis", num_ctx=num_ctx))
        vision = stack.enter_context(model_client("vision", num_ctx=num_ctx))
        return run_pipeline(
            os_client,
            answer_llm,
            analysis,
            question,
            known_docs=known,
            explicit_docs=explicit,
            top_k=top_k,
            vision=vision,
            progress=lambda name, detail: typer.echo(f"  [{name}] {detail}"),
        )


def _show_trace(trace, path=None):
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(trace.clarification or trace.answer.text or MESSAGES.get(trace.answer.refusal, ""))
    for block in trace.answer.context.blocks:
        if block.n in trace.answer.cited:
            typer.echo(f"  [{block.n}] {block.doc_id} · p.{block.page} · {block.origin}")
    if trace.error:
        typer.echo(f"  처리 오류: {trace.error}")
    total = sum(stage.seconds for stage in trace.stages)
    model_seconds = sum(event.get("seconds", 0) for event in trace.events)
    typer.echo(
        f"  [전체 {total:.2f}초 · 모델 {model_seconds:.2f}초 · "
        f"LLM {trace.calls}회 · 캐시 {trace.cache_hits}회]"
    )
    slowest = sorted(trace.stages, key=lambda stage: stage.seconds, reverse=True)[:5]
    if slowest:
        summary = " · ".join(f"{s.name} {s.seconds:.2f}초" for s in slowest)
        typer.echo(f"  [느린 단계 {summary}]")
    if trace.spans:
        labels = {
            "bedrock.chat": "생성 모델",
            "bedrock.embed": "Titan",
            "opensearch": "OpenSearch",
            "pdf": "PDF",
        }
        totals = {
            category: sum(
                float(item.get("seconds", 0))
                for item in trace.spans
                if item.get("category") == category
            )
            for category in labels
        }
        local = max(0.0, total - sum(totals.values()))
        resource_parts = [
            *(f"{labels[key]} {value:.2f}초" for key, value in totals.items()),
            f"로컬 {local:.2f}초",
        ]
        resources = " · ".join(
            resource_parts
        )
        typer.echo(f"  [시간 사용처 {resources}]")
        input_tokens = sum(int(item.get("input_tokens", 0)) for item in trace.spans)
        output_tokens = sum(int(item.get("output_tokens", 0)) for item in trace.spans)
        typer.echo(
            f"  [외부 호출 {len(trace.spans)}건 · 토큰 {input_tokens:,}→{output_tokens:,}]"
        )


@app.command()
def enhance(
    doc: Annotated[list[str] | None, typer.Option("--doc", help="보강할 문서, 반복 가능")] = None,
    report_file: Annotated[Path, typer.Option("--report")] = Path("work/enhance-report.json"),
) -> None:
    """원문·원본 벡터를 보존하고 문맥 벡터, 조건, 절 카드를 추가한다."""
    from haeindex.augmentation import enhance_document

    os_client = client()
    wanted = _resolve_docs(os_client, doc or []) or sorted(doc_counts(os_client))
    results = []
    with model_client("enrich", num_ctx=16384) as llm:
        for doc_id in wanted:
            results.append(enhance_document(os_client, llm, llm, doc_id, progress=typer.echo))
            report_file.parent.mkdir(parents=True, exist_ok=True)
            report_file.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    typer.echo(f"검색 보강 {len(results)}개 문서 · 기록: {report_file}")


@app.command()
def serve(port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8787) -> None:
    """HAEINDEX 웹 화면을 로컬에서 연다."""
    from haeindex.web import serve as serve_web

    serve_web(port)


def report_pair(
    shown: str,
    attr: str,
    base: str,
    other: str,
    arms: dict[str, list],
    iters: int,
) -> None:
    cmp = paired_bootstrap(
        shown,
        base,
        other,
        [getattr(x, attr) for x in arms[base]],
        [getattr(x, attr) for x in arms[other]],
        iters=iters,
    )
    ci = f"[{cmp.ci_low:+.3f}, {cmp.ci_high:+.3f}]"
    typer.echo(
        f"  {shown:8}{cmp.mean_a:>9.3f}{cmp.mean_b:>8.3f}{cmp.diff:>+9.3f}{ci:>20}"
        f"  {'✓' if cmp.significant else '—'}"
    )


@app.command()
def agent(
    question: Annotated[str, typer.Argument()],
    top_k: Annotated[int, typer.Option("--top-k")] = 5,
    max_searches: Annotated[int, typer.Option("--max-searches")] = 3,
    min_cos: Annotated[float, typer.Option("--min-cos")] = 0.78,
    doc_first: Annotated[bool, typer.Option("--doc-first/--doc-later")] = True,
) -> None:
    os_client = client()
    counts = doc_counts(os_client)
    with model_client("answer") as ol:
        tr = run_agent(
            os_client,
            ol,
            question,
            doc_counts=counts,
            top_k=top_k,
            max_searches=max_searches,
            min_cos=min_cos,
            doc_first=doc_first,
        )
    for s in tr.steps:
        mark = " (형식오류→폴백)" if s.parse_fail else ""
        if s.action == "search":
            where = s.doc or "전체"
            typer.echo(f'  {s.n}. 검색 "{s.query}" [{where}] → {s.n_hits}건{mark}')
        else:
            typer.echo(f"  {s.n}. {s.action}{mark}")
    ans = tr.answer
    typer.echo("")
    if ans.refusal:
        typer.echo(f"거부 [{ans.refusal}] {MESSAGES[ans.refusal]}")
    else:
        typer.echo(ans.text)
        for b in ans.context.blocks:
            if b.n in ans.cited:
                typer.echo(f"  [{b.n}] {b.doc_id} · p.{b.page} · {b.label[:52]}")
    typer.echo(
        f"\n[LLM {tr.calls}회 · 검색 {tr.searches}회 · {tr.seconds:.1f}초"
        f" · 파싱실패 {tr.parse_fails}]"
    )


@app.command("eval-agent")
def eval_agent(
    goldset: Annotated[Path, typer.Option("--goldset", exists=True)] = Path("goldset/all.yaml"),
    top_k: Annotated[int, typer.Option("--top-k")] = 5,
    max_searches: Annotated[int, typer.Option("--max-searches")] = 3,
    min_cos: Annotated[float, typer.Option("--min-cos")] = 0.78,
    iters: Annotated[int, typer.Option("--iters")] = 10000,
    doc_first: Annotated[bool, typer.Option("--doc-first/--doc-later")] = True,
) -> None:
    gold = Goldset.load(goldset)
    os_client = client()
    counts = doc_counts(os_client)
    gold.verify(list(counts))

    arms: dict[str, list] = {"agent": [], "hybrid": []}
    traces = []
    refused: list[tuple] = []
    with model_client("answer") as ol:
        for q in gold.ranked:
            tr = run_agent(
                os_client,
                ol,
                q.query,
                doc_counts=counts,
                top_k=top_k,
                max_searches=max_searches,
                min_cos=min_cos,
                doc_first=doc_first,
            )
            traces.append(tr)
            arms["agent"].append(score_query(q, [h.source for h in tr.hits[:top_k]], top_k))
            res = run_search(os_client, q.query, embedder=ol, top_k=top_k)
            arms["hybrid"].append(score_query(q, [h.source for h in res.hits], top_k))
        for q in gold.refusal:
            tr = run_agent(
                os_client,
                ol,
                q.query,
                doc_counts=counts,
                top_k=top_k,
                max_searches=max_searches,
                min_cos=min_cos,
                doc_first=doc_first,
            )
            traces.append(tr)
            refused.append((q, tr.answer.refusal, tr.searches))

    mode = "doc-first" if doc_first else "doc-later"
    typer.echo(
        f"골든셋 {goldset}  ·  순위 {len(gold.ranked)}문항  ·  검색예산 {max_searches}  ·  {mode}"
    )
    typer.echo(f"\n  {'arm':8}{'nDCG':>7}{'Recall':>8}{'MRR':>7}{'라우팅':>8}{'0히트':>7}")
    typer.echo("  " + "─" * 44)
    for a, s in ((a, summarize(a, r)) for a, r in arms.items()):
        typer.echo(
            f"  {a:8}{s.ndcg:>7.3f}{s.recall:>8.3f}{s.mrr:>7.3f}"
            f"{s.right_doc:>8.3f}{len(s.zero_hit):>7}"
        )

    summaries = {a: summarize(a, r) for a, r in arms.items()}
    typer.echo("\n  버킷별 nDCG")
    buckets = sorted(summaries["agent"].by_bucket)
    typer.echo("  " + f"{'arm':8}" + "".join(f"{b:>10}" for b in buckets))
    for a, s in summaries.items():
        typer.echo(f"  {a:8}" + "".join(f"{s.by_bucket[b]['ndcg']:>10.3f}" for b in buckets))

    typer.echo(f"\n  짝지은 부트스트랩 — agent 기준 (iters={iters}, seed=0)")
    typer.echo(f"  {'지표':8}{'agent':>9}{'hybrid':>8}{'차이':>9}{'95% CI':>20}  유의")
    typer.echo("  " + "─" * 60)
    for shown, attr in (("nDCG", "ndcg"), ("Recall", "recall"), ("MRR", "rr")):
        report_pair(shown, attr, "agent", "hybrid", arms, iters)

    ok = sum(1 for _, got, _ in refused if got is not None)
    typer.echo(f"\n  버킷 E 거부 {ok}/{len(refused)}  (임계 {min_cos})")
    for q, got, n in refused:
        if not got:
            typer.echo(f"    통과  검색{n}회  {q.id}  {q.query[:32]}")

    typer.echo("\n  비용")
    for k, v in summarize_traces(traces).items():
        typer.echo(f"    {k:14}{v:.2f}" if isinstance(v, float) else f"    {k:14}{v}")


@app.command()
def rerank(
    per_chunk: Annotated[int, typer.Option("--per-chunk")] = 2,
    lang_match: Annotated[bool, typer.Option("--lang-match/--no-lang-match")] = True,
    held_out: Annotated[list[str] | None, typer.Option("--held-out")] = None,
    l2: Annotated[float, typer.Option("--l2")] = 1.0,
    iters: Annotated[int, typer.Option("--iters")] = 600,
    out: Annotated[Path, typer.Option("--out")] = Path("work/rerank.json"),
) -> None:
    patterns = held_out or ["agentic-retrieval", "별표-3"]
    os_client = client()
    docs = sorted(doc_counts(os_client))
    held = [d for d in docs if any(p in d for p in patterns)]
    train = [d for d in docs if d not in held]
    if not train or not held:
        raise typer.BadParameter(f"분할이 비었다. 학습 {len(train)} · 검증 {len(held)}")

    typer.echo(f"학습 문서 {len(train)} · 검증 문서 {len(held)}  (문서 단위 분할)")
    for d in held:
        typer.echo(f"  검증  {d[:52]}")
    with model_client("analysis") as ol:
        tr = harvest(os_client, ol, doc_ids=train, per_chunk=per_chunk, lang_match=lang_match)
        va = harvest(os_client, ol, doc_ids=held, per_chunk=per_chunk, lang_match=lang_match)

    model = fit(
        tr.rows,
        train_docs=train,
        held_out=held,
        lang_match=lang_match,
        l2=l2,
        iters=iters,
    )
    model.save(out)

    for name, h in (("학습", tr), ("검증", va)):
        typer.echo(
            f"\n  {name}  질의 {h.n_queries} · 행 {len(h.rows)}"
            f" (양성 {sum(r.label for r in h.rows)})"
            f" · 언어불일치 버림 {h.skipped_lang} · 후보밖 {h.skipped_missing}"
        )
        base_mrr, model_mrr = mrr_of(h.rows, None), mrr_of(h.rows, model)
        typer.echo(f"        MRR  RRF {base_mrr:.3f} → 리랭커 {model_mrr:.3f}")

    typer.echo(f"\n  계수 (표준화 후 · 손실 {model.loss:.4f})")
    for name, w in model.coefficients():
        bar = "█" * min(28, int(abs(w) * 14))
        typer.echo(f"    {name:12}{w:+8.3f}  {bar}")
    typer.echo(f"\n  저장 {out}")


@app.command()
def eval(
    goldset: Annotated[Path, typer.Option("--goldset", exists=True)] = Path("goldset/all.yaml"),
    top_k: Annotated[int, typer.Option("--top-k")] = 5,
    min_cos: Annotated[float, typer.Option("--min-cos")] = 0.78,
    iters: Annotated[int, typer.Option("--iters")] = 10000,
    rerank_model: Annotated[Path | None, typer.Option("--rerank", exists=True)] = None,
    listwise: Annotated[bool, typer.Option("--listwise", help="LLM 재정렬 arm 을 넣는다")] = False,
    depth: Annotated[int, typer.Option("--depth")] = LISTWISE_DEPTH,
) -> None:
    gold = Goldset.load(goldset)
    os_client = client()
    gold.verify(list(doc_counts(os_client)))
    model = RerankModel.load(rerank_model) if rerank_model else None

    arms: dict[str, list] = {"oracle": [], "hybrid": [], "bm25": []}
    if model:
        arms["rerank"] = []
    if listwise:
        arms["listwise"] = []
    cost = {"calls": 0, "moved": 0, "parse_fails": 0}
    refused: list[tuple] = []
    with model_client("analysis") as ol:
        for q in gold.ranked:
            for arm, emb, docs in (
                ("oracle", ol, [q.doc_id]),
                ("hybrid", ol, []),
                ("bm25", None, []),
            ):
                res = run_search(os_client, q.query, embedder=emb, doc_ids=docs, top_k=top_k)
                arms[arm].append(score_query(q, [h.source for h in res.hits], top_k))
            if model or listwise:
                wide = run_search(os_client, q.query, embedder=ol, top_k=CANDIDATE_K)
            if model:
                hits = apply_rerank(model, q.query, wide.hits)[:top_k]
                arms["rerank"].append(score_query(q, [h.source for h in hits], top_k))
            if listwise:
                r = listwise_rerank(ol, q.query, wide.hits, depth=depth)
                for k in cost:
                    cost[k] += getattr(r, k)
                arms["listwise"].append(score_query(q, [h.source for h in r.hits[:top_k]], top_k))
        for q in gold.refusal:
            res = run_search(os_client, q.query, embedder=ol, top_k=top_k)
            refused.append((q, should_refuse(res, min_cos), res.top_score("knn")))

    summaries = {a: summarize(a, s) for a, s in arms.items()}
    typer.echo(f"골든셋 {goldset}  ·  순위 {len(gold.ranked)}문항  ·  top_k={top_k}")
    typer.echo(
        f"\n  {'arm':8}{'nDCG':>7}{'Recall':>8}{'MRR':>7}{'chunk_P':>9}{'라우팅':>8}{'0히트':>7}"
    )
    typer.echo("  " + "─" * 52)
    for a, s in summaries.items():
        typer.echo(
            f"  {a:8}{s.ndcg:>7.3f}{s.recall:>8.3f}{s.mrr:>7.3f}"
            f"{s.chunk_precision:>9.3f}{s.right_doc:>8.3f}{len(s.zero_hit):>7}"
        )

    typer.echo("\n  버킷별 nDCG")
    buckets = sorted(summaries["hybrid"].by_bucket)
    typer.echo("  " + f"{'arm':8}" + "".join(f"{b:>10}" for b in buckets))
    for a, s in summaries.items():
        typer.echo(f"  {a:8}" + "".join(f"{s.by_bucket[b]['ndcg']:>10.3f}" for b in buckets))

    pairs = [("hybrid", "bm25"), ("oracle", "hybrid")]
    if model:
        pairs.append(("rerank", "hybrid"))
    if listwise:
        pairs.append(("listwise", "hybrid"))
    for base, other in pairs:
        typer.echo(f"\n  짝지은 부트스트랩 — {base} 기준 (iters={iters}, seed=0)")
        typer.echo(f"  {'지표':8}{base:>9}{other:>8}{'차이':>9}{'95% CI':>20}  유의")
        typer.echo("  " + "─" * 60)
        for shown, attr in (("nDCG", "ndcg"), ("Recall", "recall"), ("MRR", "rr")):
            report_pair(shown, attr, base, other, arms, iters)

    ok = sum(1 for _, got, _ in refused if got is not None)
    typer.echo(f"\n  버킷 E 거부 {ok}/{len(refused)}  (임계 {min_cos})")
    for q, got, cos in refused:
        if not got:
            typer.echo(f"    통과  {cos:.3f}  {q.id}  {q.query[:34]}")
    if summaries["hybrid"].zero_hit:
        typer.echo(f"\n  hybrid 0히트: {', '.join(summaries['hybrid'].zero_hit)}")
    if listwise:
        n = len(gold.ranked)
        typer.echo(
            f"\n  재정렬 비용  depth {depth} · LLM {cost['calls']}회"
            f" ({cost['calls'] / n:.1f}회/질의) · 자리바뀜 {cost['moved'] / n:.1f}개/질의"
            f" · 파싱실패 {cost['parse_fails']}"
        )
        typer.echo(f"  listwise 0히트: {', '.join(summaries['listwise'].zero_hit) or '없음'}")


@app.command()
def docs(
    inbox: Annotated[Path, typer.Option("--inbox", exists=True, file_okay=False)] = Path(
        "data/inbox"
    ),
    sample: Annotated[int, typer.Option("--sample")] = 8,
) -> None:
    pdfs = sorted(inbox.glob("*.pdf"))
    if not pdfs:
        raise typer.BadParameter(f"PDF 가 없다: {inbox}/*.pdf")

    header = f"{'문서':32}{'쪽':>5}{'cid':>7}{'이중':>6}{'단':>4}"
    typer.echo(header + f"{'블록':>7}{'평균자':>7}{'단무시차':>9}{'가로지름':>9}{'본문pt':>8}")
    typer.echo("─" * 100)
    for pdf in pdfs:
        n = page_count(pdf)
        d = load_pages(pdf, _sample(n, sample))
        chars = [c for p in d for c in p.chars]
        ncol = stable_columns(column_counts(d))

        n_blocks = n_naive = n_cross = 0
        tiers: Counter[float] = Counter()
        for p in d:
            g = page_gutters(p.chars, p.width, p.height)
            bs = build_blocks(p, g)
            n_blocks += len(bs)
            n_naive += len(build_blocks(p, []))
            n_cross += sum(1 for b in build_blocks(p, []) if crosses_gutter(b, g))
            tiers += size_tiers(bs)
        body = tiers.most_common(1)[0][0] if tiers else 0.0
        avg = sum(b for b in [len(chars)]) / n_blocks if n_blocks else 0

        typer.echo(
            f"{pdf.stem[:31]:32}{n:>5}{cid_ratio(chars):>7.3f}"
            f"{doubled_ratio(chars):>6.2f}{ncol:>4}"
            f"{n_blocks:>7}{avg:>7.0f}{n_blocks - n_naive:>9}{n_cross:>9}{body:>8.1f}"
        )


def main() -> None:
    app()
