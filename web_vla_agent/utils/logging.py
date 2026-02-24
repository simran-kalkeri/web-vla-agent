"""
Structured logging utilities for the VLA Web Agent.

Provides:
- Console + rotating-file handlers
- JSON-format metric logging
- Timer context manager / decorator for profiling
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional


# ── JSON formatter ───────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        entry: Dict[str, Any] = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if hasattr(record, "metrics"):
            entry["metrics"] = record.metrics  # type: ignore[attr-defined]
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = str(record.exc_info[1])
        return json.dumps(entry)


# ── Logger factory ───────────────────────────────────────────

_CONFIGURED = False


def get_logger(
    name: str = "vla",
    level: str = "INFO",
    log_dir: str = "logs",
) -> logging.Logger:
    """Return a named logger; configure handlers on first call."""
    global _CONFIGURED

    logger = logging.getLogger(name)

    if not _CONFIGURED:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))

        # Console handler — human-readable
        ch = logging.StreamHandler()
        ch.setFormatter(
            logging.Formatter(
                "[%(asctime)s %(levelname)s %(name)s] %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(ch)

        # File handler — JSON lines (one file per run)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(
            os.path.join(log_dir, f"vla_{stamp}.jsonl"), encoding="utf-8"
        )
        fh.setFormatter(JSONFormatter())
        logger.addHandler(fh)

        _CONFIGURED = True

    return logger


# ── Metrics logger ───────────────────────────────────────────

def log_metrics(
    logger: logging.Logger,
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    prefix: str = "",
) -> None:
    """Log a metrics dict as a structured JSON record."""
    payload = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}
    if step is not None:
        payload["step"] = step
    record = logger.makeRecord(
        logger.name, logging.INFO, "", 0, "metrics", (), None
    )
    record.metrics = payload  # type: ignore[attr-defined]
    logger.handle(record)


# ── Timer ────────────────────────────────────────────────────

@contextmanager
def timer(label: str = "block", logger: Optional[logging.Logger] = None):
    """Context manager that logs elapsed wall-clock time."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    msg = f"[timer] {label}: {elapsed:.3f}s"
    if logger:
        logger.info(msg)
    else:
        print(msg)


def timed(func):
    """Decorator version of *timer*."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[timer] {func.__qualname__}: {elapsed:.3f}s")
        return result
    return wrapper
