import hmac
import logging
import signal
import threading
import time
from concurrent import futures
from hashlib import sha256
from typing import Callable

import grpc  # type: ignore

from .config import AppConfig, load_config
from .model import EmbeddingType
from .proto import transformer_pb2 as pb  # type: ignore
from .proto import transformer_pb2_grpc as pbg  # type: ignore
from .registry import ModelRegistry


# ----------------------------
# gRPC Service Implementations
# ----------------------------


class InferenceService(pbg.InferenceServiceServicer):
    def __init__(self, registry: ModelRegistry):
        self._registry = registry

    def _get_model(self, model_name: str, context):
        try:
            return self._registry.get(model_name)
        except KeyError as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))

    @staticmethod
    def _abort_inference_error(context, exc: Exception):
        if isinstance(exc, ValueError):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        if isinstance(exc, TimeoutError):
            context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))
        if isinstance(exc, RuntimeError):
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        raise exc

    def EmbedWithOffsets(self, request: pb.OffsetRequest, context):  # noqa: N802
        t0 = time.perf_counter()
        model = self._get_model(request.model_name, context)
        texts = [x.text for x in request.inputs]
        offsets = [(x.start, x.end) for x in request.inputs]
        try:
            results = model.embed(
                embedding_type=EmbeddingType.OFFSET,
                texts=texts,
                char_offsets=offsets,
            )
        except (ValueError, RuntimeError, TimeoutError) as exc:
            self._abort_inference_error(context, exc)

        response = pb.OffsetResponse()
        for group in results:
            emb = group[0]
            response.embeddings.add(string=emb.string, embedding=emb.embedding)
        logging.info(
            "EmbedWithOffsets: batch=%d took %.1fms",
            len(texts),
            (time.perf_counter() - t0) * 1e3,
        )
        return response

    def EmbedSentence(self, request: pb.SentenceRequest, context):  # noqa: N802
        t0 = time.perf_counter()
        model = self._get_model(request.model_name, context)
        texts = list(request.texts)
        try:
            results = model.embed(
                embedding_type=EmbeddingType.SENTENCE, texts=texts
            )
        except (ValueError, RuntimeError, TimeoutError) as exc:
            self._abort_inference_error(context, exc)

        response = pb.SentenceResponse()
        for group in results:
            emb = group[0]
            response.embeddings.add(string=emb.string, embedding=emb.embedding)
        logging.info(
            "EmbedSentence: batch=%d took %.1fms",
            len(texts),
            (time.perf_counter() - t0) * 1e3,
        )
        return response

    def EmbedTokens(self, request: pb.TokenRequest, context):  # noqa: N802
        t0 = time.perf_counter()
        model = self._get_model(request.model_name, context)
        texts = list(request.texts)
        try:
            results = model.embed(
                embedding_type=EmbeddingType.TOKEN, texts=texts
            )
        except (ValueError, RuntimeError, TimeoutError) as exc:
            self._abort_inference_error(context, exc)

        response = pb.TokenResponse()
        for token_list in results:
            bundle = pb.TokenEmbeddings()
            for emb in token_list:
                bundle.tokens.add(string=emb.string, embedding=emb.embedding)
            response.results.append(bundle)
        logging.info(
            "EmbedTokens: batch=%d took %.1fms",
            len(texts),
            (time.perf_counter() - t0) * 1e3,
        )
        return response


class AdminService(pbg.AdminServiceServicer):
    def __init__(self, registry: ModelRegistry):
        self._registry = registry

    def LoadModel(self, request: pb.LoadModelRequest, context):  # noqa: N802
        try:
            self._registry.load(request.model_name)
            return pb.LoadModelResponse(
                ok=True,
                message=f"Loaded {request.model_name}",
            )
        except Exception as exc:
            logging.exception("Failed to load model")
            return pb.LoadModelResponse(ok=False, message=str(exc))

    def UnloadModel(self, request: pb.UnloadModelRequest, context):  # noqa: N802
        self._registry.unload(request.model_name)
        return pb.UnloadModelResponse(
            ok=True,
            message=f"Unloaded {request.model_name}",
        )

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
            return deny(
                grpc.StatusCode.UNAUTHENTICATED,
                "Missing authentication metadata",
            )

        try:
            ts = int(timestamp)
        except ValueError:
            return deny(
                grpc.StatusCode.UNAUTHENTICATED,
                "Invalid timestamp format",
            )

        if abs(time.time() - ts) > self.ALLOWED_DRIFT_SECONDS:
            return deny(
                grpc.StatusCode.UNAUTHENTICATED,
                "Timestamp too far from current time",
            )

        expected = hmac.new(
            self.secret,
            timestamp.encode("utf-8"),
            sha256,
        ).hexdigest()

        if not hmac.compare_digest(expected, signature):
            return deny(
                grpc.StatusCode.UNAUTHENTICATED,
                "Invalid HMAC signature",
            )

        return continuation(handler_call_details)


# ----------------------------
# Server Setup & Lifecycle
# ----------------------------


def _build_server(max_workers: int, secret: str | None = None) -> grpc.Server:
    interceptors = []
    if secret:
        interceptors.append(AuthServerInterceptor(secret))
        logging.info("Security enabled")
    else:
        logging.info("Security disabled")

    return grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
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
    if threading.current_thread() is threading.main_thread():
        def _handler(signum, frame):  # noqa: ARG001
            stop_fn(signum)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)


def serve(config: AppConfig | None = None) -> None:
    # Load at runtime, not import time.  This makes GRAPHWISE_CONFIG reliable in
    # tests and allows independent server instances to use different configs.
    if config is None:
        config = load_config()

    registry = ModelRegistry(
        precision=config.precision,
        flash_attention=config.flash_attention,
        max_queue_size=config.max_queue_size,
        max_batch_tokens=config.max_batch_tokens,
        max_batch_wait_ms=config.max_batch_wait_ms,
        cpu_workers=config.cpu_workers,
        runtime_target=config.runtime_target,
        worker_start_timeout_seconds=config.worker_start_timeout_seconds,
        request_timeout_seconds=config.request_timeout_seconds,
    )
    try:
        registry.load(config.default_model)
        logging.info("Preloaded default model: %s", config.default_model)
    except Exception as exc:
        logging.warning(
            "Could not preload default model '%s': %s",
            config.default_model,
            exc,
        )

    server = _build_server(config.max_workers, config.secret)
    _register_services(server, registry)
    bound_port = server.add_insecure_port(f"[::]:{config.port}")
    if bound_port == 0:
        _unload_all_models(registry)
        raise RuntimeError(f"Could not bind gRPC server to port {config.port}")

    stop_lock = threading.Lock()
    stop_future = None

    def _graceful_stop(signum: int) -> None:
        nonlocal stop_future
        with stop_lock:
            if stop_future is not None:
                return
            logging.info("Shutting down gRPC server (signal %s)", signum)
            # Stop accepting new RPCs first and give in-flight requests time to
            # finish.  Models are unloaded only after the server has drained.
            stop_future = server.stop(grace=30)

    server.start()
    _install_signal_handlers(_graceful_stop)
    logging.info("gRPC server started on port %d", bound_port)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Interrupted, stopping server")
        _graceful_stop(signal.SIGINT)
    finally:
        with stop_lock:
            if stop_future is None:
                stop_future = server.stop(grace=30)
            local_stop_future = stop_future
        try:
            local_stop_future.wait(timeout=35)
        finally:
            _unload_all_models(registry)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        serve()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
