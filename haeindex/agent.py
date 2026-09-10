import json
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any

from opensearchpy import OpenSearch
from pydantic import BaseModel, ConfigDict, Field

from haeindex.answer import Answer, Refusal
from haeindex.answer import answer as run_answer
from haeindex.index import INDEX
from haeindex.ollama import Ollama, Truncated
from haeindex.search import Hit, Result
from haeindex.search import search as run_search

MAX_SEARCHES = 3
MAX_CALLS = 5
OBS_BODY_CHARS = 110
NUM_PREDICT = 200
JSON_OBJ = re.compile(r"\{[^{}]*\}", re.DOTALL)

SYSTEM = """당신은 문서 검색 에이전트입니다. 한 번에 **한 줄 JSON 하나만** 출력하세요.
설명·인사·코드블록을 붙이지 마세요.

가능한 행동:
{"action":"search","query":"검색어","doc":"문서 일부 이름"}   doc 은 생략 가능(전체 검색)
{"action":"answer"}    지금까지 찾은 조각으로 답할 수 있을 때
{"action":"refuse"}    문서에 답이 없다고 판단할 때

지침:
① 검색 결과가 질문과 맞지 않으면 **문서에 쓰일 만한 표현으로 질의를 바꿔** 다시 검색하세요.
② 결과가 여러 문서에 흩어져 있으면 맞는 문서 하나를 골라 `doc` 을 지정해 다시 검색하세요.
③ 답할 만한 조각이 이미 있으면 더 검색하지 말고 {"action":"answer"} 를 내세요.
④ 검색은 최대 %d회입니다."""

DOC_LATER = """
⑤ **첫 검색에는 `doc` 을 쓰지 마세요.** 전체를 검색해 결과를 본 뒤,
   답이 특정 문서에 있다는 근거가 생겼을 때만 `doc` 으로 좁히세요."""


class Step(BaseModel):
    model_config = ConfigDict(frozen=True)

    n: int
    action: str
    query: str = ""
    doc: str = ""
    n_hits: int = 0
    raw: str = ""
    parse_fail: bool = False
    doc_ignored: bool = False


class Trace(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str
    steps: list[Step] = Field(default_factory=list)
    hits: list[Hit] = Field(default_factory=list)
    answer: Answer = Answer()
    calls: int = 0
    searches: int = 0
    parse_fails: int = 0
    seconds: float = 0.0
    truncated: bool = False

    def result(self, top_k: int) -> Result:
        return Result(query=self.question, hits=self.hits[:top_k])


def parse_action(raw: str) -> dict[str, str] | None:
    for m in JSON_OBJ.finditer(raw):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "action" in obj:
            return {k: str(v) for k, v in obj.items()}
    return None


def match_doc(needle: str, doc_ids: Sequence[str]) -> list[str]:
    if not needle:
        return []
    low = needle.strip().lower()
    hit = [d for d in doc_ids if low in d.lower()]
    if hit:
        return hit
    parts = [p for p in re.split(r"[\s_·/-]+", low) if len(p) >= 2]
    return [d for d in doc_ids if any(p in d.lower() for p in parts)]


def render_docs(counts: Mapping[str, int]) -> str:
    return "\n".join(f"- {d} ({n}조각)" for d, n in sorted(counts.items()))


def render_obs(step: Step, hits: Sequence[Hit]) -> str:
    where = step.doc or "전체"
    lines = [f'검색 "{step.query}" (문서={where}) → {len(hits)}건']
    for i, h in enumerate(hits, 1):
        s = h.source
        cos = h.legs["knn"].score if "knn" in h.legs else 0.0
        label = str(s.get("path") or s.get("title") or "")
        doc = str(s.get("doc_id", ""))[:14]
        lines.append(f"{i}. [{doc}] p{s.get('page')} {label[:46]} (코사인 {cos:.2f})")
        lines.append(f"   {str(s.get('body', ''))[:OBS_BODY_CHARS].replace(chr(10), ' ')}")
    return "\n".join(lines)


def merge_hits(existing: Sequence[Hit], fresh: Sequence[Hit]) -> list[Hit]:
    by_id = {h.chunk_id: h for h in existing}
    for h in fresh:
        old = by_id.get(h.chunk_id)
        if old is None or h.fused > old.fused:
            by_id[h.chunk_id] = h
    return sorted(by_id.values(), key=lambda h: (-h.fused, h.chunk_id))


def run(
    os_client: OpenSearch,
    ol: Ollama,
    question: str,
    *,
    doc_counts: Mapping[str, int],
    top_k: int = 5,
    max_searches: int = MAX_SEARCHES,
    max_calls: int = MAX_CALLS,
    min_cos: float = 0.78,
    index: str = INDEX,
    strict: bool = True,
    doc_first: bool = True,
) -> Trace:
    started = time.monotonic()
    doc_ids = sorted(doc_counts)
    rules = SYSTEM % max_searches if doc_first else (SYSTEM % max_searches) + DOC_LATER
    system = rules + "\n\n색인된 문서:\n" + render_docs(doc_counts)
    convo: list[dict[str, str]] = [{"role": "user", "content": f"질문: {question}"}]
    steps: list[Step] = []
    hits: list[Hit] = []
    calls = searches = parse_fails = 0
    truncated = False
    decision = "answer"

    while searches < max_searches and calls < max_calls - 1:
        try:
            raw = ol.chat(
                [{"role": "system", "content": system}, *convo], num_predict=NUM_PREDICT
            ).content
        except Truncated:
            truncated = True
            raw = ""
        calls += 1
        act = parse_action(raw)
        fail = act is None
        if fail:
            act = (
                {"action": "search", "query": raw.strip()[:120] or question}
                if not hits
                else {"action": "answer"}
            )
            parse_fails += 1

        action = act.get("action", "answer")
        if action != "search":
            steps.append(Step(n=len(steps) + 1, action=action, raw=raw[:200], parse_fail=fail))
            decision = action
            break

        query = act.get("query", "").strip() or question
        wanted = match_doc(act.get("doc", ""), doc_ids)
        ignored = bool(wanted) and not doc_first and searches == 0
        if ignored:
            wanted = []
        res = run_search(
            os_client,
            query,
            embedder=ol,
            doc_ids=wanted,
            top_k=top_k,
            index=index,
        )
        searches += 1
        step = Step(
            n=len(steps) + 1,
            action="search",
            query=query,
            doc=act.get("doc", "").strip(),
            n_hits=len(res.hits),
            raw=raw[:200],
            parse_fail=fail,
            doc_ignored=ignored,
        )
        steps.append(step)
        hits = merge_hits(hits, res.hits)
        convo.append({"role": "assistant", "content": json.dumps(act, ensure_ascii=False)})
        convo.append({"role": "user", "content": render_obs(step, res.hits)})

    merged = Result(query=question, hits=hits[:top_k])
    if decision == "refuse" and not hits:
        ans = Answer(refusal=Refusal.NO_HITS)
    else:
        try:
            ans = run_answer(ol, merged, question, min_cos=min_cos, strict=strict)
            calls += 1
        except Truncated:
            truncated = True
            ans = Answer(refusal=Refusal.NO_CITATIONS)
    return Trace(
        question=question,
        steps=steps,
        hits=hits,
        answer=ans,
        calls=calls,
        searches=searches,
        parse_fails=parse_fails,
        seconds=time.monotonic() - started,
        truncated=truncated,
    )


def summarize_traces(traces: Sequence[Trace]) -> dict[str, Any]:
    n = len(traces) or 1
    return {
        "질문": len(traces),
        "LLM호출/질문": sum(t.calls for t in traces) / n,
        "검색/질문": sum(t.searches for t in traces) / n,
        "초/질문": sum(t.seconds for t in traces) / n,
        "파싱실패": sum(t.parse_fails for t in traces),
        "문서지정": sum(1 for t in traces if any(s.doc for s in t.steps)),
        "재검색": sum(1 for t in traces if t.searches > 1),
        "문서지정무시": sum(1 for t in traces for s in t.steps if s.doc_ignored),
        "잘림": sum(1 for t in traces if t.truncated),
    }
