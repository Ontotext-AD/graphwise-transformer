"""Request ID propagation and request-aware logging.

The request ID comes from the ``x-request-id`` gRPC metadata key (gRPC's
equivalent of the ``X-Request-ID`` HTTP header). It is kept in a
:class:`contextvars.ContextVar`, so every log record emitted while serving a
request, in the server process or in a model worker process, carries it as
``%(request_id)s``.
"""

from __future__ import annotations

import contextvars
import logging
import re
import uuid
from collections.abc import Iterable

REQUEST_ID_METADATA_KEY = "x-request-id"
NO_REQUEST_ID = "-"

# Accept the usual ID shapes (UUIDs, hex, ULIDs, dotted/colon-separated trace
# IDs) and reject anything else so client input cannot forge log lines.
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:\-]{1,128}$")

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=NO_REQUEST_ID
)

_factory_installed = False


def new_request_id() -> str:
    return uuid.uuid4().hex


def normalize_request_id(value: str | bytes | None) -> str:
    """Return a client-supplied request ID if it is safe to log, else a new one."""
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="replace")
    if value:
        value = value.strip()
        if _VALID_REQUEST_ID.match(value):
            return value
    return new_request_id()


def request_id_from_metadata(
    metadata: Iterable[tuple[str, str | bytes]] | None,
) -> str:
    for key, value in metadata or ():
        if key.lower() == REQUEST_ID_METADATA_KEY:
            return normalize_request_id(value)
    return new_request_id()


def join_request_ids(request_ids: Iterable[str]) -> str:
    """Label for log lines that cover several requests (a worker microbatch)."""
    unique = list(dict.fromkeys(request_ids))
    return ",".join(unique) if unique else NO_REQUEST_ID


def install_record_factory() -> None:
    """Add ``request_id`` to every LogRecord, whichever logger creates it.

    A record factory, unlike a filter, also covers records from child and
    third-party loggers, so ``%(request_id)s`` never raises a KeyError.
    """
    global _factory_installed
    if _factory_installed:
        return
    previous = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get()
        return record

    logging.setLogRecordFactory(factory)
    _factory_installed = True

