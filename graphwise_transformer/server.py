import logging
import os
import signal
import base64
import hmac
import time
from concurrent import futures
from typing import Callable
from hashlib import sha256
from grpc import ServerInterceptor, StatusCode

import grpc  # type: ignore

from .config import load_config
from .registry import ModelRegistry
from .model import EmbeddingType
from .proto import transformer_pb2 as pb  # type: ignore
from .proto import transformer_pb2_grpc as pbg  # type: ignore

CONFIG = load_config(os.environ.get("GRAPHWISE_CONFIG"))

# ----------------------------
# gRPC Service Implementations
# ----------------------------
class InferenceService(pbg.InferenceServiceServicer):
    def __init__(self, registry: ModelRegistry):
        self._registry = registry

    def EmbedWithOffsets(self, request: pb.OffsetRequest, context):  # noqa: N802 (gRPC name)
        model = self._registry.get(request.model_name)
        texts = [x.text for x in request.inputs]
        offsets = [(x.start, x.end) for x in request.inputs]
        results = model.embed(embedding_type=EmbeddingType.OFFSET, texts=texts, char_offsets=offsets)

        response = pb.OffsetResponse()
        for group in results:
            emb = group[0]
            response.embeddings.add(string=emb.string, embedding=emb.embedding)
        return response

    def EmbedSentence(self, request: pb.SentenceRequest, context):  # noqa: N802
        model = self._registry.get(request.model_name)
        results = model.embed(embedding_type=EmbeddingType.SENTENCE, texts=list(request.texts))

        response = pb.SentenceResponse()
        for group in results:
            emb = group[0]
            response.embeddings.add(string=emb.string, embedding=emb.embedding)
        return response

    def EmbedTokens(self, request: pb.TokenRequest, context):  # noqa: N802
        model = self._registry.get(request.model_name)
        results = model.embed(embedding_type=EmbeddingType.TOKEN, texts=list(request.texts))

        response = pb.TokenResponse()
        for token_list in results:
            bundle = pb.TokenEmbeddings()
            for emb in token_list:
                bundle.tokens.add(string=emb.string, embedding=emb.embedding)
            response.results.append(bundle)
        return response


class AdminService(pbg.AdminServiceServicer):
    def __init__(self, registry: ModelRegistry):
        self._registry = registry

    def LoadModel(self, request: pb.LoadModelRequest, context):  # noqa: N802
        try:
            self._registry.load(request.model_name)
            return pb.LoadModelResponse(ok=True, message=f"Loaded {request.model_name}")
        except Exception as exc:
            logging.exception("Failed to load model")
            return pb.LoadModelResponse(ok=False, message=str(exc))

    def UnloadModel(self, request: pb.UnloadModelRequest, context):  # noqa: N802
        self._registry.unload(request.model_name)
        return pb.UnloadModelResponse(ok=True, message=f"Unloaded {request.model_name}")

    def ListModels(self, request: pb.ListModelsRequest, context):  # noqa: N802
        names = self._registry.list_models()
        return pb.ListModelsResponse(model_names=names)

class AuthServerInterceptor(grpc.ServerInterceptor):
    ALLOWED_DRIFT_SECONDS = 30

    def __init__(self, secret: str):
        self.secret = secret.encode("utf-8")

    def intercept_service(self, continuation, handler_call_details):
        metadata = dict(handler_call_details.invocation_metadata or [])
        timestamp = metadata.get("x-timestamp")
        signature = metadata.get("x-signature")

        def deny(code, message):
            def abort_handler(request, context):
                context.abort(code, message)
            return grpc.unary_unary_rpc_method_handler(abort_handler)

        if not timestamp or not signature:
            return deny(grpc.StatusCode.UNAUTHENTICATED, "Missing authentication metadata")

        try:
            ts = int(timestamp)
        except ValueError:
            return deny(grpc.StatusCode.UNAUTHENTICATED, "Invalid timestamp format")

        if abs(time.time() - ts) > self.ALLOWED_DRIFT_SECONDS:
            return deny(grpc.StatusCode.UNAUTHENTICATED, "Timestamp too far from current time")

        expected = hmac.new(self.secret, timestamp.encode("utf-8"), sha256).hexdigest()

        if not hmac.compare_digest(expected, signature):
            return deny(grpc.StatusCode.UNAUTHENTICATED, "Invalid HMAC signature")

        return continuation(handler_call_details)

# ----------------------------
# Server Setup & Lifecycle
# ----------------------------

def _build_server(max_workers: int) -> grpc.Server:
    interceptors = []
    if CONFIG.secret:
        interceptors.append(AuthServerInterceptor(CONFIG.secret))
        logging.info("Security enabled")
    else:
        logging.info("Security disabled")
    return grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers),
                       interceptors=interceptors if interceptors else None,
                       )


def _register_services(server: grpc.Server, registry: ModelRegistry) -> None:
    pbg.add_InferenceServiceServicer_to_server(InferenceService(registry), server)
    pbg.add_AdminServiceServicer_to_server(AdminService(registry), server)


def _unload_all_models(registry: ModelRegistry) -> None:
    for model_name in registry.list_models():
        try:
            registry.unload(model_name)
        except Exception as unload_err:  # pragma: no cover
            logging.warning("Error unloading model %s: %s", model_name, unload_err)


def _install_signal_handlers(stop_fn: Callable[[int], None]) -> None:
    """Install signal handlers only if running in the main thread."""
    import threading
    if threading.current_thread() is threading.main_thread():
        def _handler(signum, frame):  # noqa: ARG001
            stop_fn(signum)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)


def serve() -> None:
    registry = ModelRegistry()
    try:
        registry.load(CONFIG.default_model)
        logging.info("Preloaded default model: %s", CONFIG.default_model)
    except Exception as exc:
        logging.warning("Could not preload default model '%s': %s", CONFIG.default_model, exc)

    server = _build_server(CONFIG.max_workers)
    _register_services(server, registry)
    server.add_insecure_port(f"[::]:{CONFIG.port}")

    def _graceful_stop(signum: int) -> None:
        logging.info("Shutting down gRPC server (signal %s)", signum)
        try:
            _unload_all_models(registry)
        finally:
            server.stop(0)

    _install_signal_handlers(_graceful_stop)

    server.start()
    logging.info("gRPC server started on port %d", CONFIG.port)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Interrupted, stopping server")
        _graceful_stop(signal.SIGINT)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        serve()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
