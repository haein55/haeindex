"""요청 단위의 가벼운 wall-time 프로파일러.

ContextVar를 사용해 Bedrock·검색·PDF 하위 함수가 파이프라인 인자를 오염시키지 않고
현재 요청에 span을 남긴다. 모든 작업은 현재 단일 요청 스레드에서 순차 실행된다.
"""

import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

_active: ContextVar["Profiler | None"] = ContextVar("haeindex_profiler", default=None)
_task: ContextVar[str] = ContextVar("haeindex_profile_task", default="")


@dataclass
class Profiler:
    stage: str = "startup"
    spans: list[dict[str, Any]] = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)

    def activate(self) -> Token:
        return _active.set(self)

    @staticmethod
    def deactivate(token: Token) -> None:
        _active.reset(token)


def set_task(name: str) -> Token:
    return _task.set(name)


def reset_task(token: Token) -> None:
    _task.reset(token)


@contextmanager
def span(category: str, name: str, **metadata: Any):
    """현재 프로파일러에 leaf 작업 하나를 기록한다. 비활성 상태에서는 no-op이다."""

    profiler = _active.get()
    details = dict(metadata)
    if profiler is None:
        yield details
        return
    started = time.monotonic()
    stage = profiler.stage
    task = _task.get()
    try:
        yield details
    except Exception as exc:
        details.setdefault("error", type(exc).__name__)
        raise
    finally:
        event = {
            "category": category,
            "name": name,
            "stage": stage,
            "at": round(started - profiler.started, 6),
            "seconds": round(time.monotonic() - started, 6),
        }
        if task:
            event["task"] = task
        event.update(details)
        profiler.spans.append(event)
