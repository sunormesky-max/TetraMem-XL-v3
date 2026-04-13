"""
Structured logging and distributed tracing for TetraMem-XL.

Provides JSON-formatted structured logs with trace_id propagation
for cross-bucket operation tracking.
"""

import json
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Optional


_trace_stack: list = []


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def current_trace_id() -> str:
    return _trace_stack[-1] if _trace_stack else ""


@contextmanager
def trace_context(trace_id: Optional[str] = None):
    tid = trace_id or new_trace_id()
    _trace_stack.append(tid)
    try:
        yield tid
    finally:
        if _trace_stack:
            _trace_stack.pop()


class StructuredLogger:
    def __init__(self, name: str):
        self._logger = logging.getLogger(name)
        self._name = name

    def _log(self, level: str, event: str, **kwargs: Any) -> None:
        record = {
            "timestamp": time.time(),
            "level": level,
            "logger": self._name,
            "event": event,
            "trace_id": current_trace_id(),
        }
        record.update(kwargs)
        msg = json.dumps(record, default=str)
        getattr(self._logger, level.lower())(msg)

    def info(self, event: str, **kwargs: Any) -> None:
        self._log("INFO", event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._log("WARNING", event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._log("ERROR", event, **kwargs)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._log("DEBUG", event, **kwargs)
