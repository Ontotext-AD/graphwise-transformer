import logging
import queue
import threading

import grpc
import pytest

from graphwise_transformer import model as model_mod
from graphwise_transformer.model import EmbeddingModel, EmbeddingType, _WorkItem
from graphwise_transformer.proto import transformer_pb2 as pb
from graphwise_transformer.proto import transformer_pb2_grpc as pbg
from graphwise_transformer.request_context import (
    install_record_factory,
    join_request_ids,
    normalize_request_id,
    request_id_from_metadata,
    request_id_var,
)
from graphwise_transformer.server import InferenceService, _build_server
from tests.test_server import _FakeRegistry


@pytest.fixture(autouse=True)
def _request_id_logging():
    install_record_factory()


# ----------------------------
# Request ID normalization
# ----------------------------


@pytest.mark.parametrize(
    "value",
    ["abc-123", "0b1f6d3c-5f3e-4a7e-9d0c-2f1d8f7a6b5c", "trace:span.1_x"],
)
def test_valid_request_id_is_kept(value):
    assert normalize_request_id(value) == value
    assert normalize_request_id(f"  {value} ".encode()) == value


@pytest.mark.parametrize(
    "value", [None, "", "   ", "a\nINFO forged line", "x" * 129, "id with spaces"]
)
def test_missing_or_unsafe_request_id_is_replaced(value):
    result = normalize_request_id(value)
    assert result != value
    assert len(result) == 32


def test_join_request_ids_deduplicates_in_order():
    assert join_request_ids(["b", "a", "b"]) == "b,a"
    assert join_request_ids([]) == "-"


# ----------------------------
# Server interceptor
# ----------------------------


def _start_server(secret=None):
    server = _build_server(2, secret)
    pbg.add_InferenceServiceServicer_to_server(
        InferenceService(_FakeRegistry()), server
    )
    port = server.add_insecure_port("localhost:0")
    server.start()
    return server, grpc.insecure_channel(f"localhost:{port}")


def _rpc_records(caplog, prefix):
    return [r for r in caplog.records if r.getMessage().startswith(prefix)]


def test_request_id_header_is_logged_and_echoed(caplog):
    server, channel = _start_server()
    try:
        stub = pbg.InferenceServiceStub(channel)
        with caplog.at_level(logging.INFO):
            _response, call = stub.EmbedSentence.with_call(
                pb.SentenceRequest(model_name="model", texts=["a"]),
                metadata=(("x-request-id", "req-42"),),
            )
        assert ("x-request-id", "req-42") in call.initial_metadata()
        records = _rpc_records(caplog, "EmbedSentence")
        assert [r.request_id for r in records] == ["req-42"]
    finally:
        channel.close()
        server.stop(None)


def test_missing_request_id_is_generated_and_echoed(caplog):
    server, channel = _start_server()
    try:
        stub = pbg.InferenceServiceStub(channel)
        with caplog.at_level(logging.INFO):
            _response, call = stub.EmbedTokens.with_call(
                pb.TokenRequest(model_name="model", texts=["a"])
            )
        echoed = dict(call.initial_metadata())["x-request-id"]
        assert len(echoed) == 32
        assert [r.request_id for r in _rpc_records(caplog, "EmbedTokens")] == [echoed]
    finally:
        channel.close()
        server.stop(None)


def test_request_id_does_not_leak_between_calls(caplog):
    server, channel = _start_server()
    try:
        stub = pbg.InferenceServiceStub(channel)
        with caplog.at_level(logging.INFO):
            for rid in ("first", "second", "third"):
                stub.EmbedSentence(
                    pb.SentenceRequest(model_name="model", texts=["a"]),
                    metadata=(("x-request-id", rid),),
                )
        records = _rpc_records(caplog, "EmbedSentence")
        assert [r.request_id for r in records] == ["first", "second", "third"]
    finally:
        channel.close()
        server.stop(None)


def test_missing_model_is_logged_with_request_id(caplog):
    server, channel = _start_server()
    try:
        stub = pbg.InferenceServiceStub(channel)
        with caplog.at_level(logging.INFO), pytest.raises(grpc.RpcError) as err:
            stub.EmbedSentence(
                pb.SentenceRequest(model_name="missing", texts=["a"]),
                metadata=(("x-request-id", "missing-1"),),
            )
        assert err.value.code() == grpc.StatusCode.NOT_FOUND
        records = _rpc_records(caplog, "NOT_FOUND")
        assert [r.request_id for r in records] == ["missing-1"]
    finally:
        channel.close()
        server.stop(None)


def test_rejected_auth_is_logged_with_request_id(caplog):
    server, channel = _start_server(secret="s3cret")
    try:
        stub = pbg.InferenceServiceStub(channel)
        with caplog.at_level(logging.INFO), pytest.raises(grpc.RpcError) as err:
            stub.EmbedSentence(
                pb.SentenceRequest(model_name="model", texts=["a"]),
                metadata=(("x-request-id", "unauth-1"),),
            )
        assert err.value.code() == grpc.StatusCode.UNAUTHENTICATED
        records = _rpc_records(caplog, "Rejected")
        assert [r.request_id for r in records] == ["unauth-1"]
    finally:
        channel.close()
        server.stop(None)


def test_python_client_rejects_uppercase_metadata_key(caplog):
    """gRPC metadata keys are HTTP/2 header names, which must be lowercase.

    The Python client does not lowercase them for you: an uppercase key fails
    the call on the client before anything is sent, so callers must use
    ``x-request-id``, not ``X-Request-ID``.
    """
    server, channel = _start_server()
    try:
        stub = pbg.InferenceServiceStub(channel)
        with caplog.at_level(logging.INFO), pytest.raises(grpc.RpcError) as err:
            stub.EmbedSentence(
                pb.SentenceRequest(model_name="model", texts=["a"]),
                metadata=(("X-Request-ID", "upper-1"),),
            )
        assert err.value.code() == grpc.StatusCode.INTERNAL
        assert err.value.details() == "Invalid metadata"
        assert _rpc_records(caplog, "EmbedSentence") == []  # never reached the server

        # The same call with the lowercase key succeeds.
        _response, call = stub.EmbedSentence.with_call(
            pb.SentenceRequest(model_name="model", texts=["a"]),
            metadata=(("x-request-id", "upper-1"),),
        )
        assert ("x-request-id", "upper-1") in call.initial_metadata()
    finally:
        channel.close()
        server.stop(None)


@pytest.mark.parametrize("key", ["x-request-id", "X-Request-ID", "X-REQUEST-ID"])
def test_server_matches_request_id_key_case_insensitively(key):
    # Clients that lowercase keys for you (Java, Go, HTTP/2 proxies) always send
    # lowercase; the server still accepts any casing it might be handed.
    assert request_id_from_metadata([(key, "abc-1")]) == "abc-1"


# ----------------------------
# Propagation into model workers
# ----------------------------


class _ImmediateQueue:
    """Stands in for the worker input queue and answers every item at once."""

    def __init__(self, owner):
        self.owner = owner
        self.items = []

    def put(self, item, timeout=None):  # noqa: ARG002
        self.items.append(item)
        self.owner._pending[item.request_id].put(
            model_mod._WorkResult(request_id=item.request_id, results=[])
        )


def test_embed_stamps_work_item_with_current_request_id():
    pool = object.__new__(EmbeddingModel)
    pool._closed = False
    pool._pending = {}
    pool._pending_lock = threading.Lock()
    pool._request_timeout_seconds = 1.0
    pool._input_queue = _ImmediateQueue(pool)

    token = request_id_var.set("client-7")
    try:
        pool.embed(EmbeddingType.SENTENCE, texts=["a"])
    finally:
        request_id_var.reset(token)

    (item,) = pool._input_queue.items
    assert item.trace_id == "client-7"
    assert item.request_id != "client-7"  # internal routing key is separate


class _FailingModel:
    def encode(self, *args, **kwargs):
        raise RuntimeError("boom")


def _run_worker(monkeypatch, items):
    monkeypatch.setattr(model_mod, "configure_logging", lambda settings: None)
    monkeypatch.setattr(
        model_mod,
        "_load_sentence_transformer",
        lambda *a, **k: (_FailingModel(), "fp32", "sdpa"),
    )
    monkeypatch.setattr(
        model_mod, "_tokenize_batch", lambda m, texts: [[1, 2] for _ in texts]
    )
    input_queue, result_queue, ready_queue = queue.Queue(), queue.Queue(), queue.Queue()
    for item in items:
        input_queue.put(item)
    input_queue.put(None)
    model_mod._worker_main(
        "fake", "cpu", "auto", False, 16384, 200.0,
        input_queue, result_queue, ready_queue, None,
    )
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    return results


def test_worker_logs_batch_failure_with_all_request_ids(monkeypatch, caplog):
    items = [
        _WorkItem("k1", EmbeddingType.SENTENCE.value, ("a",), None, "req-a"),
        _WorkItem("k2", EmbeddingType.SENTENCE.value, ("b",), None, "req-b"),
    ]
    with caplog.at_level(logging.INFO):
        results = _run_worker(monkeypatch, items)

    assert {r.request_id for r in results} == {"k1", "k2"}
    assert all(r.error_type == "RuntimeError" for r in results)
    failures = _rpc_records(caplog, "Inference failed")
    assert [r.request_id for r in failures] == ["req-a,req-b"]
    assert request_id_var.get() == "-"


def test_worker_logs_tokenization_failure_with_item_request_id(monkeypatch, caplog):
    def tokenize(model, texts):
        if texts == ("bad",):
            raise ValueError("cannot tokenize")
        return [[1, 2] for _ in texts]

    items = [_WorkItem("k1", EmbeddingType.SENTENCE.value, ("bad",), None, "req-bad")]
    monkeypatch.setattr(model_mod, "configure_logging", lambda settings: None)
    with caplog.at_level(logging.INFO):
        monkeypatch.setattr(model_mod, "_tokenize_batch", tokenize)
        monkeypatch.setattr(
            model_mod,
            "_load_sentence_transformer",
            lambda *a, **k: (_FailingModel(), "fp32", "sdpa"),
        )
        input_queue, result_queue = queue.Queue(), queue.Queue()
        for item in items:
            input_queue.put(item)
        input_queue.put(None)
        model_mod._worker_main(
            "fake", "cpu", "auto", False, 16384, 1.0,
            input_queue, result_queue, queue.Queue(), None,
        )

    assert result_queue.get().error_type == "ValueError"
    failures = _rpc_records(caplog, "Tokenization failed")
    assert [r.request_id for r in failures] == ["req-bad"]
