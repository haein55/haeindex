"""로컬 문서 대화 UI. 정적 자산과 작업 API를 같은 loopback 서버에서 제공한다."""

import io
import json
import threading
import time
from contextlib import ExitStack, suppress
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit
from uuid import uuid4

import pdfplumber

from haeindex.answer import MESSAGES
from haeindex.augmentation import enhance_document
from haeindex.bedrock import bedrock_embedding_model, bedrock_model
from haeindex.index import INDEX, client, doc_counts, ensure_index, index_chunks
from haeindex.load_pdf import page_count
from haeindex.models import model_client
from haeindex.paths import slugify
from haeindex.pipeline import run_pipeline
from haeindex.profile import Profile, build
from haeindex.source_recovery import pdf_for

STATIC = Path(__file__).with_name("static")
MAX_UPLOAD = 32 * 1024 * 1024
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/cat.svg": ("cat.svg", "image/svg+xml"),
    "/cat-black.svg": ("cat-black.svg", "image/svg+xml"),
    "/cat-cheese.svg": ("cat-cheese.svg", "image/svg+xml"),
    "/cat-tabby.svg": ("cat-tabby.svg", "image/svg+xml"),
    "/cat-munchkin.svg": ("cat-munchkin.svg", "image/svg+xml"),
}


class BusyError(ValueError):
    pass


@dataclass
class Job:
    id: str = field(default_factory=lambda: uuid4().hex)
    status: str = "running"
    events: list[dict] = field(default_factory=list)
    result: dict | None = None
    error: str = ""
    started: float = field(default_factory=time.monotonic)
    elapsed: float | None = None


class Application:
    def __init__(self, inbox: Path = Path("data/inbox")):
        self.inbox = inbox.resolve()
        self.lock = threading.Lock()
        self.jobs: dict[str, Job] = {}
        self.active: str | None = None

    def documents(self) -> dict:
        os_client = client()
        try:
            counts = doc_counts(os_client)
        finally:
            os_client.close()
        docs = []
        for doc_id, count in sorted(counts.items()):
            path = pdf_for(doc_id, self.inbox)
            try:
                pages = Profile.load(doc_id).n_pages
            except (FileNotFoundError, ValueError):
                pages = None
            docs.append(
                {
                    "id": doc_id,
                    "name": path.name if path else doc_id,
                    "chunks": count,
                    "pages": pages,
                    "has_pdf": path is not None,
                }
            )
        return {"documents": docs}

    def health(self) -> dict:
        answer_model = bedrock_model("answer")
        status = {
            "search": False,
            "answer_model": answer_model,
            "analysis_model": bedrock_model("analysis"),
            "vision_model": bedrock_model("vision"),
            "embed_model": bedrock_embedding_model(),
            "missing_models": [],
            "provider": "bedrock",
            "embed_provider": "bedrock",
        }
        os_client = client()
        try:
            status["search"] = bool(os_client.ping())
        finally:
            os_client.close()
        with self.lock:
            status["active_job"] = self.active
        status["models"] = not status["missing_models"]
        return status

    def submit(self, task, *args) -> str:
        with self.lock:
            if self.active:
                raise BusyError("다른 작업을 처리 중입니다. 완료 후 다시 시도해 주세요.")
            while len(self.jobs) >= 30:
                self.jobs.pop(next(iter(self.jobs)))
            job = Job()
            self.jobs[job.id] = job
            self.active = job.id

        def progress(name: str, detail: str):
            with self.lock:
                now = time.monotonic()
                offset = now - job.started
                if job.events:
                    job.events[-1]["seconds"] = round(offset - job.events[-1]["at"], 3)
                job.events.append(
                    {"name": name, "detail": detail, "at": round(offset, 3), "seconds": 0.0}
                )
                job.events[:] = job.events[-200:]

        def run():
            try:
                result = task(*args, progress=progress)
                with self.lock:
                    job.result, job.status = result, "complete"
            except Exception as exc:
                with self.lock:
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.status = "error"
            finally:
                with self.lock:
                    job.elapsed = time.monotonic() - job.started
                    if job.events:
                        job.events[-1]["seconds"] = round(
                            job.elapsed - job.events[-1]["at"], 3
                        )
                    self.active = None

        threading.Thread(target=run, daemon=True, name=f"haeindex-{job.id[:8]}").start()
        return job.id

    def job(self, job_id: str) -> dict | None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            events = [dict(event) for event in job.events]
            elapsed = job.elapsed if job.elapsed is not None else time.monotonic() - job.started
            if events and job.elapsed is None:
                events[-1]["seconds"] = round(elapsed - events[-1]["at"], 3)
            return {
                "id": job.id,
                "status": job.status,
                "events": events,
                "result": job.result,
                "error": job.error,
                "seconds": round(
                    elapsed, 1
                ),
            }

    def ask(self, question: str, documents: list[str], *, progress) -> dict:
        with ExitStack() as stack:
            os_client = client()
            stack.callback(os_client.close)
            known = sorted(doc_counts(os_client))
            if any(d not in known for d in documents):
                raise ValueError("선택한 문서가 색인에 없습니다. 문서 목록을 새로고침해 주세요.")
            if not known:
                raise ValueError("먼저 PDF를 추가해 주세요.")
            answer = stack.enter_context(model_client("answer", num_ctx=16384))
            analysis = stack.enter_context(model_client("analysis", num_ctx=16384))
            vision = stack.enter_context(model_client("vision", num_ctx=16384))
            trace = run_pipeline(
                os_client,
                answer,
                analysis,
                question,
                known_docs=known,
                explicit_docs=documents,
                vision=vision,
                progress=progress,
            )
        result = trace.model_dump(mode="json")
        result["message"] = (
            trace.clarification
            or trace.answer.text
            or MESSAGES.get(trace.answer.refusal, "답변을 작성하지 못했습니다.")
        )
        # UI/API는 실제로 인용한 원문 블록만 출처 목록에 제시한다.
        result["citations"] = [
            b.model_dump() for b in trace.answer.context.blocks if b.n in trace.answer.cited
        ]
        return result

    def enhance(self, documents: list[str], *, progress) -> dict:
        with ExitStack() as stack:
            os_client = client()
            stack.callback(os_client.close)
            known = doc_counts(os_client)
            if not documents or any(d not in known for d in documents):
                raise ValueError("보강할 문서를 선택해 주세요.")
            llm = stack.enter_context(model_client("enrich", num_ctx=16384))
            reports = [
                enhance_document(
                    os_client, llm, llm, doc, progress=lambda detail: progress("enhancing", detail)
                )
                for doc in documents
            ]
        failures = sum(len(r["failures"]) for r in reports)
        return {
            "message": f"문서 {len(reports)}개 검색 보강 완료 · 실패 {failures}건",
            "reports": reports,
        }

    def upload(self, filename: str, data: bytes, *, progress) -> dict:
        # 기존 PDF/색인을 덮어쓰지 않는다. slug 충돌도 별도로 검사한다.
        if Path(filename).name != filename or "\\" in filename or "\x00" in filename:
            raise ValueError("올바른 PDF 파일명이 아닙니다.")
        if not filename.lower().endswith(".pdf") or not data.startswith(b"%PDF-"):
            raise ValueError("PDF 파일만 추가할 수 있습니다.")
        doc_id = slugify(Path(filename))
        progress("reading", "PDF를 확인하고 있습니다")
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            if not pdf.pages:
                raise ValueError("비어 있는 PDF입니다.")
        self.inbox.mkdir(parents=True, exist_ok=True)
        destination = self.inbox / f"{Path(filename).stem}.pdf"
        with ExitStack() as stack:
            os_client = client()
            stack.callback(os_client.close)
            if (
                destination.exists()
                or pdf_for(doc_id, self.inbox)
                or doc_id in doc_counts(os_client)
            ):
                raise ValueError("같은 이름의 문서가 있습니다. 파일명을 변경해 추가해 주세요.")
            with destination.open("xb") as handle:
                handle.write(data)
            progress("profiling", "문서 구조와 읽는 순서를 분석합니다")
            prof = build(destination, head=page_count(destination))
            prof.save()
            from haeindex.cli import _chunks_for

            _, chunks, _ = _chunks_for(destination, prof, 1200)
            if not chunks:
                raise ValueError("추출할 텍스트가 없습니다. OCR 처리된 PDF를 사용해 주세요.")
            progress("embedding", f"원문 청크 {len(chunks)}개를 검색에 연결합니다")
            ol = stack.enter_context(model_client("enrich", num_ctx=16384))
            vectors = ol.embed_batched([c.text for c in chunks])
            ensure_index(os_client)
            ok, errors = index_chunks(os_client, chunks, vectors)
            os_client.indices.refresh(index=INDEX)
            if errors:
                raise RuntimeError(
                    f"청크 {ok}개 저장, {len(errors)}개 실패. CLI로 재색인해 주세요."
                )
            report = enhance_document(
                os_client, ol, ol, doc_id, progress=lambda detail: progress("enhancing", detail)
            )
        return {"message": f"{filename} 문서를 추가했습니다.", "doc_id": doc_id, "report": report}


class WebServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, application=None):
        self.application = application or Application()
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server: WebServer

    def log_message(self, fmt, *args):
        # 질문, 파일명, 응답 원문을 HTTP 로그에 남기지 않는다.
        pass

    def _send(self, status: int, data: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; script-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
        )
        self.end_headers()
        with suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(data)

    def _json(self, status: int, data: dict):
        self._send(
            status, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8"
        )

    def _local_request(self) -> bool:
        host = self.headers.get("Host", "")
        port = self.server.server_port
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        origin = self.headers.get("Origin")
        if host not in allowed or (origin and origin not in {f"http://{h}" for h in allowed}):
            self._json(403, {"error": "로컬 화면에서 접속해 주세요."})
            return False
        return True

    def do_GET(self):
        if not self._local_request():
            return
        url = urlsplit(self.path)
        app = self.server.application
        try:
            if url.path in ASSETS:
                filename, mime = ASSETS[url.path]
                self._send(200, (STATIC / filename).read_bytes(), mime)
            elif url.path == "/api/documents":
                self._json(200, app.documents())
            elif url.path == "/api/health":
                self._json(200, app.health())
            elif url.path.startswith("/api/jobs/"):
                job = app.job(url.path.rsplit("/", 1)[-1])
                self._json(200 if job else 404, job or {"error": "작업을 찾을 수 없습니다."})
            elif url.path == "/api/pdf":
                doc_id = parse_qs(url.query).get("doc", [""])[0]
                path = pdf_for(doc_id, app.inbox)
                if path is None:
                    self._json(404, {"error": "원본 PDF를 찾을 수 없습니다."})
                else:
                    self._send(200, path.read_bytes(), "application/pdf")
            else:
                self._json(404, {"error": "페이지를 찾을 수 없습니다."})
        except Exception as exc:
            self._json(503, {"error": f"연결을 확인해 주세요. {type(exc).__name__}: {exc}"})

    def do_POST(self):
        if not self._local_request():
            return
        app = self.server.application
        try:
            length = int(self.headers.get("Content-Length", "0"))
            limit = MAX_UPLOAD if self.path == "/api/upload" else 16384
            if not 0 < length <= limit:
                self._json(413, {"error": "요청 크기를 확인해 주세요. PDF는 최대 32MB입니다."})
                return
            self.connection.settimeout(30)
            content_type = self.headers.get("Content-Type", "").split(";")[0]
            if self.path == "/api/upload":
                if content_type != "application/pdf":
                    raise ValueError("PDF 파일만 추가할 수 있습니다.")
                filename = unquote(self.headers.get("X-Filename", ""))
                job_id = app.submit(app.upload, filename, self.rfile.read(length))
            elif self.path in {"/api/ask", "/api/enhance"}:
                if content_type != "application/json":
                    raise ValueError("JSON 요청이 필요합니다.")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("요청은 객체여야 합니다.")
                docs = body.get("documents", [])
                if (
                    not isinstance(docs, list)
                    or len(docs) > 20
                    or any(not isinstance(d, str) or len(d) > 100 for d in docs)
                ):
                    raise ValueError("문서 선택을 확인해 주세요.")
                if self.path == "/api/ask":
                    question = body.get("question", "")
                    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 2000:
                        raise ValueError("질문은 1~2,000자로 입력해 주세요.")
                    job_id = app.submit(app.ask, question.strip(), docs)
                else:
                    job_id = app.submit(app.enhance, docs)
            else:
                self._json(404, {"error": "경로를 찾을 수 없습니다."})
                return
            self._json(202, {"id": job_id})
        except BusyError as exc:
            self._json(409, {"error": str(exc)})
        except (ValueError, TimeoutError) as exc:
            self._json(400, {"error": str(exc)})


def serve(port: int = 8787) -> None:
    with WebServer(("127.0.0.1", port)) as server:
        print(f"HAEINDEX → http://127.0.0.1:{server.server_port}  (종료: Ctrl+C)", flush=True)
        with suppress(KeyboardInterrupt):
            server.serve_forever()
