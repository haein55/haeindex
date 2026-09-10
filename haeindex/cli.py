from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from haeindex.agent import run as run_agent
from haeindex.agent import summarize_traces
from haeindex.answer import MESSAGES, should_refuse
from haeindex.answer import answer as run_answer
from haeindex.blocks import build_blocks, crosses_gutter, page_blocks, size_tiers
from haeindex.chunks import (
    build_sections,
    chunk_fixed,
    chunk_sections,
    report,
)
from haeindex.evaluate import Goldset, paired_bootstrap, score_query, summarize
from haeindex.features import gen_many
from haeindex.geometry import column_counts, page_gutters, stable_columns
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
from haeindex.load_pdf import (
    cid_ratio,
    doubled_ratio,
    load_pages,
    page_count,
    top_fonts,
)
from haeindex.ollama import Ollama
from haeindex.paths import slugify
from haeindex.profile import Profile, build
from haeindex.rerank import Model as RerankModel
from haeindex.rerank import fit, harvest, mrr_of
from haeindex.rerank import rerank as apply_rerank
from haeindex.search import CANDIDATE_K
from haeindex.search import search as run_search

app = typer.Typer(add_completion=False, no_args_is_help=True, help="PDF → 검색 → 답변")


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
) -> None:
    n = page_count(pdf)
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
) -> None:
    n = page_count(pdf)
    docs_ = load_pages(pdf, _sample(n, sample))
    counts = column_counts(docs_, header=header, footer=footer, bins=bins)
    typer.echo(f"{pdf.name}  (표본 {len(docs_)}쪽, bins={bins}, 밴드 {header}~{footer})")
    typer.echo(f"  페이지별      {_fmt(counts)}")
    typer.echo(f"  안정된 단 수   {stable_columns(counts)}")


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
    blocks = []
    for p in load_pages(pdf, list(range(1, prof.head_pages + 1))):
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
    with_queries: Annotated[
        bool, typer.Option("--with-queries", help="가상 질문 생성. 청크당 LLM 1회")
    ] = False,
) -> None:
    prof = Profile.load(slugify(pdf))
    _, chs, how = _chunks_for(pdf, prof, max_chars)
    if not chs:
        raise typer.BadParameter("청크가 0개다")

    os_client = client()
    ensure_index(os_client)

    gen: list[list[str]] | None = None
    failed = 0
    with Ollama() as ol:
        vectors = ol.embed_batched([c.text for c in chs])
        if with_queries:
            with typer.progressbar(length=len(chs), label="가상 질문") as bar:
                gen, failed = gen_many(ol, [c.body for c in chs], lambda i, n: bar.update(1))

    gone = replace_doc(os_client, prof.doc_id)
    ok, errors = index_chunks(os_client, chs, vectors, queries=gen)

    typer.echo(f"{prof.source}  방법 {how}")
    typer.echo(f"  청크 {len(chs)}개 · 임베딩 {len(vectors)}개 (dim {len(vectors[0])})")
    typer.echo(
        f"  기존 청크 {gone}건 교체 → 색인 {ok}건" + (f" · 오류 {len(errors)}" if errors else "")
    )
    if with_queries:
        typer.echo(f"  가상 질문 생성 실패 {failed}/{len(chs)} (빈 값으로 기록했다)")
    typer.echo(f"  인덱스 전체: {doc_counts(os_client)}")


@app.command()
def search(
    query: Annotated[str, typer.Argument()],
    doc: Annotated[list[str] | None, typer.Option("--doc", help="반복 가능")] = None,
    top_k: Annotated[int, typer.Option("--top-k")] = 5,
    explain: Annotated[bool, typer.Option("--explain/--no-explain")] = True,
    no_embed: Annotated[bool, typer.Option("--no-embed")] = False,
) -> None:
    os_client = client()
    ol = None if no_embed else Ollama()
    try:
        res = run_search(os_client, query, embedder=ol, doc_ids=doc or [], top_k=top_k)
    finally:
        if ol:
            ol.close()

    typer.echo(f"query={query!r}  후보 {res.candidates}  dedupe -{res.dropped}")
    if res.degraded:
        typer.echo(f"  강등된 leg: {res.degraded}")
    for i, h in enumerate(res.hits, 1):
        s = h.source
        legs = " ".join(f"{k}={v.rank}/{v.score:.1f}" for k, v in sorted(h.legs.items()))
        typer.echo(f"  {i}. [{s['doc_id'][:16]}] p{s['page']:>3} {legs}")
        typer.echo(f"     {(s.get('path') or s.get('title') or '')[:64]}")
        if explain:
            typer.echo(f"     {s.get('body', '')[:100]!r}")


@app.command()
def ask(
    question: Annotated[str, typer.Argument()],
    doc: Annotated[list[str] | None, typer.Option("--doc")] = None,
    top_k: Annotated[int, typer.Option("--top-k")] = 5,
    min_cos: Annotated[float, typer.Option("--min-cos")] = 0.78,
    num_ctx: Annotated[int, typer.Option("--num-ctx")] = 8192,
    strict: Annotated[bool, typer.Option("--strict/--no-strict")] = True,
) -> None:
    os_client = client()
    with Ollama(num_ctx=num_ctx) as ol:
        res = run_search(os_client, question, embedder=ol, doc_ids=doc or [], top_k=top_k)
        ans = run_answer(ol, res, question, min_cos=min_cos, strict=strict)

    if ans.refusal is not None:
        typer.echo(MESSAGES[ans.refusal])
        typer.echo(f"  [코사인 top1 {res.top_score('knn'):.3f} · 임계 {min_cos}]")
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
    with Ollama() as ol:
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
    with Ollama() as ol:
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
    with Ollama() as ol:
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
) -> None:
    gold = Goldset.load(goldset)
    os_client = client()
    gold.verify(list(doc_counts(os_client)))
    model = RerankModel.load(rerank_model) if rerank_model else None

    arms: dict[str, list] = {"oracle": [], "hybrid": [], "bm25": []}
    if model:
        arms["rerank"] = []
    refused: list[tuple] = []
    with Ollama() as ol:
        for q in gold.ranked:
            for arm, emb, docs in (
                ("oracle", ol, [q.doc_id]),
                ("hybrid", ol, []),
                ("bm25", None, []),
            ):
                res = run_search(os_client, q.query, embedder=emb, doc_ids=docs, top_k=top_k)
                arms[arm].append(score_query(q, [h.source for h in res.hits], top_k))
            if model:
                wide = run_search(os_client, q.query, embedder=ol, top_k=CANDIDATE_K)
                hits = apply_rerank(model, q.query, wide.hits)[:top_k]
                arms["rerank"].append(score_query(q, [h.source for h in hits], top_k))
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
