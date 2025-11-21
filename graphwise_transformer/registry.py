import threading
from typing import Dict

from .model import EmbeddingModel


class ModelRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._models: Dict[str, EmbeddingModel] = {}

    def get(self, model_name: str) -> EmbeddingModel:
        with self._lock:
            model = self._models.get(model_name)
            if model is None:
                raise KeyError(f"Model not loaded: {model_name}")
            return model

    def load(self, model_name: str) -> None:
        with self._lock:
            if model_name in self._models:
                return
            self._models[model_name] = EmbeddingModel(model_name)

    def unload(self, model_name: str) -> None:
        with self._lock:
            model = self._models.pop(model_name, None)
            if model is not None:
                model.unload()

    def list_models(self) -> list[str]:
        with self._lock:
            return list(self._models.keys())
