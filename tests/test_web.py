import io
import json
import threading
from types import SimpleNamespace

import pytest

from haeindex.web import MAX_UPLOAD, Application, BusyError, Handler


def handler(path, *, headers=None, body=b"", app=None):
    h = object.__new__(Handler)
    h.path = path
    h.headers = {
        "Host": "127.0.0.1:8787",
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
        **(headers or {}),
    }
    h.server = SimpleNamespace(server_port=8787, application=app or Application())
    h.rfile = io.BytesIO(body)
    h.connection = SimpleNamespace(settimeout=lambda _: None)
    h.responses = []
    h._json = lambda status, data: h.responses.append((status, data))
    h._send = lambda status, data, mime: h.responses.append((status, data, mime))
    return h


def test_foreign_origin_and_dns_rebinding_host_are_blocked():
    for headers in [{"Origin": "https://foreign.test"}, {"Host": "attacker.test:8787"}]:
        h = handler("/api/ask", headers=headers, body=b"{}")
        h.do_POST()
        assert h.responses[0][0] == 403


def test_invalid_question_and_document_types_do_not_start_jobs():
    for payload in [
        {"question": []},
        {"question": "x", "documents": "all"},
        {"question": "x" * 2001},
        [],
    ]:
        h = handler("/api/ask", body=json.dumps(payload).encode())
        h.do_POST()
        assert h.responses[0][0] == 400
        assert not h.server.application.jobs


def test_large_upload_and_wrong_mime_rejected_before_read():
    h = handler("/api/upload", headers={"Content-Length": str(MAX_UPLOAD + 1)})
    h.do_POST()
    assert h.responses[0][0] == 413
    h = handler("/api/upload", body=b"garbage")
    h.do_POST()
    assert h.responses[0][0] == 400


def test_unknown_file_and_asset_cannot_read_local_paths(tmp_path):
    app = Application(tmp_path)
    for path in ["/../../pyproject.toml", "/api/pdf?doc=../../pyproject.toml", "/api/jobs/missing"]:
        h = handler(path, app=app)
        h.do_GET()
        assert h.responses[0][0] == 404


def test_static_assets_are_served_without_external_dependencies():
    h = handler("/")
    h.do_GET()
    assert h.responses[0][0] == 200
    assert b'<html lang="ko">' in h.responses[0][1]


def test_only_one_model_job_and_errors_are_observable():
    app = Application()
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def task(*, progress):
        progress("testing", "working")
        entered.set()
        release.wait(2)
        raise ValueError("test failure")

    job_id = app.submit(task)
    assert entered.wait(1)
    with pytest.raises(BusyError):
        app.submit(task)
    assert app.job(job_id)["events"][0]["name"] == "testing"
    assert app.job(job_id)["events"][0]["seconds"] >= 0
    release.set()
    # Joining this bounded worker avoids races without polling/sleep loops.
    for thread in threading.enumerate():
        if thread.name == f"haeindex-{job_id[:8]}":
            thread.join(timeout=2)
            finished.set()
    assert app.job(job_id)["status"] == "error"
    assert "test failure" in app.job(job_id)["error"]
    assert app.active is None


def test_upload_rejects_paths_and_non_pdf_before_writing(tmp_path):
    app = Application(tmp_path)
    for name, data in [
        ("..\\escape.pdf", b"%PDF-"),
        ("../escape.pdf", b"%PDF-"),
        ("bad.pdf", b"not PDF"),
        ("bad.txt", b"%PDF-"),
    ]:
        with pytest.raises(ValueError):
            app.upload(name, data, progress=lambda *a: None)
    assert list(tmp_path.iterdir()) == []


def text_pdf():
    """한 페이지 텍스트 PDF. 원문 추출부터 업로드 색인까지 검증한다."""
    stream = b"BT /F1 14 Tf 30 160 Td (Application deadline: 2024-06-01.) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    data, offsets = b"%PDF-1.4\n", [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data += str(i).encode() + b" 0 obj\n" + obj + b"\nendobj\n"
    xref = len(data)
    data += b"xref\n0 6\n0000000000 65535 f \n"
    for offset in offsets[1:]:
        data += f"{offset:010} 00000 n \n".encode()
    return data + f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()


def test_pdf_upload_extracts_real_text_before_indexing(tmp_path, monkeypatch):
    from haeindex import web

    monkeypatch.chdir(tmp_path)
    os_client = SimpleNamespace(
        close=lambda: None, indices=SimpleNamespace(refresh=lambda **k: None)
    )
    monkeypatch.setattr(web, "client", lambda: os_client)
    monkeypatch.setattr(web, "doc_counts", lambda c: {})
    monkeypatch.setattr(web, "ensure_index", lambda c: None)
    captured = []

    def index_chunks(client, chunks, vectors):
        captured.extend(chunks)
        return len(chunks), []

    monkeypatch.setattr(web, "index_chunks", index_chunks)
    monkeypatch.setattr(web, "enhance_document", lambda *a, **k: {"failures": []})

    class Embedder:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def embed_batched(self, texts):
            return [[0.1] * 1024 for _ in texts]

    monkeypatch.setattr(web, "model_client", lambda *args, **kwargs: Embedder(**kwargs))
    app = Application(tmp_path / "inbox")
    result = app.upload("deadline.pdf", text_pdf(), progress=lambda *a: None)
    assert result["doc_id"] == "deadline"
    assert captured and "2024-06-01" in captured[0].body
    assert (tmp_path / "inbox" / "deadline.pdf").read_bytes() == text_pdf()
    with pytest.raises(ValueError, match="같은 이름"):
        app.upload("deadline.pdf", text_pdf(), progress=lambda *a: None)
