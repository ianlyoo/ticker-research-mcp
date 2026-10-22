"""Silence httpx/httpcore request logging while a provider request is in flight (URLs can carry identities)."""
from __future__ import annotations

import logging
from contextvars import ContextVar

_PRIVATE_HTTP = ContextVar("ticker_private_http", default=False)


class _PrivateHTTPFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _PRIVATE_HTTP.get()
