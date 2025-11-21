import datetime
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np  # type: ignore

# torch is optional at runtime; used for CUDA cache cleanup if present
try:  # type: ignore
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from sentence_transformers import SentenceTransformer  # type: ignore


class EmbeddingType(str, Enum):
    SENTENCE = "SENTENCE"
    OFFSET = "OFFSET"
    TOKEN = "TOKEN"


@dataclass
class Embedding:
    string: str
    embedding: list[float]


class EmbeddingModel:
    def __init__(self, model_name: Optional[str] = None):
        if not model_name:
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self._model_name = model_name
        self.model: Optional[SentenceTransformer] = SentenceTransformer(model_name)

    def unload(self) -> None:
        # Move model to CPU first (if torch and model is on GPU), then delete refs and collect
        try:
            if self.model is not None:
                try:
                    # For ST, underlying modules are torch.nn.Modules
                    if hasattr(self.model, "to"):
                        self.model.to("cpu")  # type: ignore[attr-defined]
                except Exception:
                    pass
                # Drop strong reference to free memory
                self.model = None
        finally:
            # Force garbage collection to promptly release CPU RAM
            try:
                import gc  # local import to avoid global state

                gc.collect()
            except Exception:
                pass
            # If CUDA is available, clear cached GPU memory
            try:
                if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():  # type: ignore[attr-defined]
                    torch.cuda.empty_cache()  # type: ignore[attr-defined]
            except Exception:
                pass

    def embed(
        self,
        embedding_type: EmbeddingType = EmbeddingType.OFFSET,
        texts: list[str] | None = None,
        char_offsets: list[tuple[int, int]] | None = None,
    ) -> list[list[Embedding]]:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        if texts is None:
            raise ValueError("texts must be provided")

        start = datetime.datetime.now()
        # Build encode kwargs - always use output_value=None to get full output dict
        # Note: We don't need return_offsets_mapping from encode() because offset_mapping
        # is obtained directly from the tokenizer in the OFFSET code path below
        encode_kwargs = {
            "output_value": None,
            "convert_to_numpy": True,
            "convert_to_tensor": False,
            "show_progress_bar": False,
        }

        output = self.model.encode(texts, **encode_kwargs)
        logging.debug(f"encoding took: {datetime.datetime.now() - start}")

        if embedding_type == EmbeddingType.SENTENCE:
            return [
                [
                    Embedding(
                        embedding=list(emb) if not isinstance(emb := x["sentence_embedding"].detach().cpu().numpy().tolist(), list) else emb,
                        string=self.model.tokenizer.decode(x["input_ids"]),
                    )
                ]
                for x in output
            ]

        if embedding_type == EmbeddingType.TOKEN:
            return [
                [
                    Embedding(
                        embedding=list(e) if not isinstance(e, list) else e,
                        string=s
                    )
                    for e, s in zip(
                        x["token_embeddings"].detach().cpu().numpy().tolist(),
                        self.model.tokenizer.convert_ids_to_tokens(x["input_ids"]),
                    )
                ]
                for x in output
            ]

        # Require explicit OFFSET type; otherwise raise
        if embedding_type != EmbeddingType.OFFSET:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

        start = datetime.datetime.now()
        if char_offsets is None or len(char_offsets) != len(texts):
            raise ValueError(
                "Offsets must be provided and must have the same length as the input in EmbeddingType.OFFSET "
            )

        offset_mappings = self.model.tokenizer(
            texts, return_offsets_mapping=True
        ).data["offset_mapping"]

        target_token_offsets: list[tuple[int, int]] = []

        for sent_oms, char_offset_tuple in zip(offset_mappings, char_offsets):
            s = e = None
            for token_idx, token_oms in enumerate(sent_oms):
                if s is None and token_oms[1] > char_offset_tuple[0]:
                    s = token_idx
                if e is None and token_oms[0] >= char_offset_tuple[1]:
                    e = token_idx
                # if there is already a start, stop if you reach a special token (e.g. [sep])
                if e is None and s is not None and token_oms[0] == token_oms[1] == 0:
                    e = token_idx
                if s is not None and e is not None:
                    break
            if s is None:
                s = 0
            if e is None or e <= s:
                e = min(s + 1, len(sent_oms))  # ensure non-empty span
            target_token_offsets.append((s, e))

        offset_embeddings: list[np.ndarray] = []
        input_strings: list[str] = []

        for x, token_ids in zip(output, target_token_offsets):
            token_embeddings = x["token_embeddings"][token_ids[0] : token_ids[1]]
            offset_embeddings.append(
                np.mean([t.detach().cpu().numpy() for t in token_embeddings], axis=0)
            )
            input_ids = x["input_ids"][token_ids[0] : token_ids[1]]
            input_strings.append(self.model.tokenizer.decode(input_ids))

        logging.debug(f"postprocessing took: {datetime.datetime.now() - start}")

        return [
            [
                Embedding(
                    embedding=list(emb) if not isinstance(emb := e.tolist(), list) else emb,
                    string=s
                )
            ]
            for e, s in zip(offset_embeddings, input_strings)
        ]
