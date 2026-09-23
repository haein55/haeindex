"""누락된 인접 청크와 원본 PDF의 문자·표·이미지를 필요할 때 보완한다."""

import base64
import hashlib
import io
from collections.abc import Sequence
from pathlib import Path

import pdfplumber

from haeindex.bedrock import Bedrock
from haeindex.index import INDEX, SOURCE_EXCLUDE
from haeindex.llm_tasks import TaskFailure, TaskRunner
from haeindex.paths import slugify
from haeindex.profiling import span
from haeindex.reasoning import Record
from haeindex.search import Hit, LegHit


def pdf_for(doc_id: str, inbox: Path = Path("data/inbox")) -> Path | None:
    return next((p for p in inbox.glob("*.pdf") if slugify(p) == doc_id), None)


def fetch_chunks(
    os_client, ids: Sequence[str], doc_ids: Sequence[str], index: str = INDEX
) -> list[Hit]:
    if not ids:
        return []
    filters = [{"terms": {"chunk_id": list(ids)}}]
    if doc_ids:
        filters.append({"terms": {"doc_id": list(doc_ids)}})
    with span("opensearch", "절 원문 가져오기", index=index) as metrics:
        rows = os_client.search(
            index=index,
            body={
                "size": min(len(ids), 60),
                "_source": {"excludes": SOURCE_EXCLUDE},
                "query": {"bool": {"filter": filters}},
            },
        )["hits"]["hits"]
        metrics["hits"] = len(rows)
    return [
        Hit(
            chunk_id=h["_source"]["chunk_id"],
            fused=0,
            source=h["_source"],
            legs={"section": LegHit(rank=i + 1, score=0)},
        )
        for i, h in enumerate(rows)
    ]


def neighbors(os_client, hits: Sequence[Hit], index: str = INDEX) -> list[Hit]:
    clauses = []
    for h in hits[:4]:
        seq = h.source.get("seq")
        if seq is not None:
            clauses.append(
                {
                    "bool": {
                        "filter": [
                            {"term": {"doc_id": h.source["doc_id"]}},
                            {"range": {"seq": {"gte": max(0, seq - 1), "lte": seq + 1}}},
                        ]
                    }
                }
            )
    if not clauses:
        return []
    with span("opensearch", "인접 원문 가져오기", index=index) as metrics:
        rows = os_client.search(
            index=index,
            body={
                "size": 16,
                "_source": {"excludes": SOURCE_EXCLUDE},
                "query": {"bool": {"should": clauses, "minimum_should_match": 1}},
            },
        )["hits"]["hits"]
        metrics["hits"] = len(rows)
    return [
        Hit(
            chunk_id=h["_source"]["chunk_id"],
            fused=0,
            source=h["_source"],
            legs={"neighbor": LegHit(rank=i + 1, score=0)},
        )
        for i, h in enumerate(rows)
    ]


class PageReading(Record):
    transcription: str
    readable: bool


class PageReview(Record):
    approved: bool
    reason: str


def recover_pages(
    hits: Sequence[Hit],
    *,
    runner: TaskRunner | None = None,
    vision: Bedrock | None = None,
    max_pages: int = 3,
    use_vision: bool = False,
    question: str = "",
) -> list[Hit]:
    pairs = list(
        dict.fromkeys(
            (str(h.source.get("doc_id", "")), int(h.source.get("page", 0)))
            for h in hits
            if not h.source.get("evidence_origin")
            or (use_vision and h.source.get("evidence_origin") == "pdf_text")
        )
    )[:max_pages]
    recovered = []
    for doc_id, number in pairs:
        path = pdf_for(doc_id)
        if path is None or number < 1:
            continue
        with pdfplumber.open(path) as pdf:
            with span("pdf", "PDF 텍스트·표 추출", document=doc_id, page=number):
                if number > len(pdf.pages):
                    continue
                page = pdf.pages[number - 1].dedupe_chars()
                text = page.extract_text(layout=False) or ""
                tables = page.extract_tables()
                if tables:
                    text += "\n\nPDF 표의 행(열 구분: |)\n" + "\n\n".join(
                        "\n".join(
                            " | ".join((c or "").replace("\n", " ") for c in row)
                            for row in table
                        )
                        for table in tables
                    )
            origin = "pdf_text"
            image_hash = ""
            if use_vision and runner is not None and vision is not None:
                buf = io.BytesIO()
                with span("pdf", "PDF 페이지 이미지 변환", document=doc_id, page=number):
                    page.to_image(resolution=160).original.save(buf, format="PNG")
                png = buf.getvalue()
                image_hash = hashlib.sha256(png).hexdigest()
                images = [base64.b64encode(png).decode()]
                try:
                    reading = runner.run(
                        vision,
                        "pdf-vision-read",
                        "PDF 이미지의 표와 본문을 읽으세요. "
                        "질문의 답을 추측하지 말고 내용을 전사하세요. "
                        "표는 각 행에 상위 열 제목·행 제목·단위·조건을 적어 관계를 보존하세요. "
                        "흐릿하거나 해석할 수 없는 부분은 제외하고 readable=false로 표시하세요. "
                        "숫자·미만/이상·각주·예외를 원본 그대로 유지하세요.",
                        {"doc_id": doc_id, "page": number, "question": question},
                        PageReading,
                        num_predict=2400,
                        images=images,
                    )
                    if reading.readable and reading.transcription.strip():
                        check = runner.run(
                            vision,
                            "pdf-vision-check",
                            "이미지와 전사를 대조하세요. 모든 숫자·연도·행/열·단위·각주가 "
                            "이미지와 일치하고 임의로 보완한 부분이 없을 때만 approved=true. "
                            "다른 행을 합치거나 조건을 누락했다면 false로 반환하세요.",
                            {"transcription": reading.transcription},
                            PageReview,
                            images=images,
                            num_predict=400,
                        )
                        if check.approved:
                            text = reading.transcription
                            origin = "vision_transcription"
                except TaskFailure:
                    pass
        if not text.strip():
            continue
        # 긴 페이지는 원문 순서를 유지한 조각으로 나누되 행 중간을 자르지 않는다.
        pieces, lines, count = [], [], 0
        for line in text.splitlines():
            if lines and count + len(line) > 4800:
                pieces.append("\n".join(lines))
                lines, count = [], 0
            lines.append(line)
            count += len(line) + 1
        if lines:
            pieces.append("\n".join(lines))
        for i, body in enumerate(pieces):
            cid = f"{doc_id}#pdf{number:04d}-{i}-{origin}"
            recovered.append(
                Hit(
                    chunk_id=cid,
                    fused=0,
                    source={
                        "chunk_id": cid,
                        "doc_id": doc_id,
                        "page": number,
                        "end_page": number,
                        "title": f"원본 PDF p.{number}",
                        "path": "",
                        "body": body,
                        "text": body,
                        "evidence_origin": origin,
                        "image_sha256": image_hash,
                    },
                )
            )
    return recovered
