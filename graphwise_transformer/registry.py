import threading
from typing import Dict

from .model import EmbeddingModel


class ModelRegistry:
    def __init__(
        self,
        *,
        precision: str = "auto",
        flash_attention: bool = True,
        max_queue_size: int = 1024,
        max_batch_tokens: int = 16384,
        max_batch_wait_ms: float = 2.0,
        cpu_workers: int = 1,
        runtime_target: str = "auto",
        worker_start_timeout_seconds: float = 300.0,
        request_timeout_seconds: float = 300.0,
    ) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._models: Dict[str, EmbeddingModel] = {}
        self._loading: set[str] = set()
        self._kwargs = {
            "precision": precision,
            "flash_attention": flash_attention,
            "max_queue_size": max_queue_size,
            "max_batch_tokens": max_batch_tokens,
            "max_batch_wait_ms": max_batch_wait_ms,
            "cpu_workers": cpu_workers,
            "runtime_target": runtime_target,
            "worker_start_timeout_seconds": worker_start_timeout_seconds,
            "request_timeout_seconds": request_timeout_seconds,
        }

    def get(self, model_name: str) -> EmbeddingModel:
        with self._condition:
            model = self._models.get(model_name)
            if model is None:
                raise KeyError(f"Model not loaded: {model_name}")
            return model

    def load(self, model_name: str) -> None:
        # Do not hold the registry state lock while workers download/load a
        # model. Existing models remain available for inference during loading.
        with self._condition:
            while model_name in self._loading:
                self._condition.wait()
            if model_name in self._models:
                return
            self._loading.add(model_name)

        try:
            model = EmbeddingModel(model_name, **self._kwargs)
        except Exception:
            with self._condition:
                self._loading.discard(model_name)
                self._condition.notify_all()
            raise

        with self._condition:
            self._models[model_name] = model
            self._loading.discard(model_name)
            self._condition.notify_all()

    def unload(self, model_name: str) -> None:
        with self._condition:
            while model_name in self._loading:
                self._condition.wait()
            model = self._models.pop(model_name, None)
        if model is not None:
            model.unload()

    def list_models(self) -> list[str]:
        with self._condition:
            return list(self._models.keys())
