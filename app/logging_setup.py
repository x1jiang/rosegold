"""Process-wide logging configuration.

Idempotent: only installs a handler when the root logger has none, so an
embedding application (or uvicorn's own logging config) keeps control.
``ROSEGOLD_LOG_LEVEL`` selects the level (default INFO); ``ROSEGOLD_LOG_JSON=1``
switches to one-JSON-object-per-line, which Cloud Logging / Loki parse natively.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict

_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True
    level_name = os.getenv("ROSEGOLD_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        if os.getenv("ROSEGOLD_LOG_JSON", "").lower() in {"1", "true", "yes"}:
            handler.setFormatter(_JsonFormatter())
        else:
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
    root.setLevel(level)
    logging.getLogger("rosegold").setLevel(level)
