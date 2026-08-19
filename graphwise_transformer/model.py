from __future__ import annotations

import importlib.util
import logging
import multiprocessing as mp
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover - import error is surfaced when loading a worker
    torch = None  # type: ignore[assignment]


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_MAX_BATCH_INPUTS = 256


class EmbeddingType(str, Enum):
    SENTENCE = "SENTENCE"
    OFFSET = "OFFSET"
    TOKEN = "TOKEN"


@dataclass
class Embedding:
    string: str
    embedding: list[float]


@dataclass(frozen=True)
class _WorkItem:
    request_id: str
    embedding_type: str
    texts: tuple[str, ...]
    char_offsets: tuple[tuple[int, int], ...] | None


@dataclass(frozen=True)
class _WorkResult:
    request_id: str
    results: list[list[Embedding]] | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class _WorkerReady:
    device: str
    precision: str
    attention: str
    error: str | None = None


@dataclass(frozen=True)
class _TokenOutput:
    token_embeddings: Any
    input_ids: list[int]


def _resolve_precision(device: str, requested: str) -> tuple[str, Any]:
    import torch as worker_torch

    requested = requested.lower()
    if requested not in {"auto", "fp16", "bf16", "fp32"}:
        raise ValueError(f"Unsupported precision: {requested}")

    if device == "cpu":
        resolved = "fp32" if requested == "auto" else requested
    elif requested == "auto":
        resolved = "bf16" if worker_torch.cuda.is_bf16_supported() else "fp16"
    else:
        resolved = requested

    if resolved == "bf16" and device != "cpu" and not worker_torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 was requested but the CUDA device does not support bf16")

    dtype = {
        "fp16": worker_torch.float16,
        "bf16": worker_torch.bfloat16,
        "fp32": worker_torch.float32,
    }[resolved]
    return resolved, dtype


def _flash_attention_device_supported(device: str) -> bool:
    """Use stable FA2 only where upstream currently documents support."""
    if device == "cpu":
        return False
    import torch as worker_torch

    index = int(device.split(":", 1)[1])
    major, _minor = worker_torch.cuda.get_device_capability(index)
    # Stable FlashAttention-2 documents Ampere/Ada/Hopper support. Newer
    # architectures safely use PyTorch SDPA until upstream FA2 documents them.
    return major in {8, 9}


def _load_sentence_transformer(
    model_name: str,
    device: str,
    precision: str,
    flash_attention: bool,
):
    import torch as worker_torch
    from sentence_transformers import SentenceTransformer

    if device.startswith("cuda:"):
        worker_torch.cuda.set_device(int(device.split(":", 1)[1]))

    resolved_precision, dtype = _resolve_precision(device, precision)
    fa_installed = importlib.util.find_spec("flash_attn") is not None
    want_fa = (
        flash_attention
        and fa_installed
        and _flash_attention_device_supported(device)
    )

    def load(attention: str):
        model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs={
                "torch_dtype": dtype,
                "attn_implementation": attention,
            },
        )
        model.eval()
        if attention == "flash_attention_2":
            # SentenceTransformers >=5 with Transformers >=5 auto-detects
            # variable-length input unpadding. Keep auto mode explicit here.
            transformer_module = model[0]
            if hasattr(transformer_module, "unpad_inputs"):
                transformer_module.unpad_inputs = None
        return model

    attention = "flash_attention_2" if want_fa else "sdpa"
    try:
        model = load(attention)
    except Exception:
        if attention != "flash_attention_2":
            raise
        logging.exception(
            "FlashAttention-2 initialization failed for %s on %s; falling back to SDPA",
            model_name,
            device,
        )
        attention = "sdpa"
        model = load(attention)

    if flash_attention and not fa_installed and device != "cpu":
        logging.warning(
            "flash_attention=true but flash-attn is not installed; using SDPA on %s",
            device,
        )
    elif flash_attention and fa_installed and not _flash_attention_device_supported(device):
        logging.warning(
            "flash-attn is installed but stable FA2 is not enabled for %s; using SDPA",
            device,
        )

    logging.info(
        "Loaded %s on %s precision=%s attention=%s",
        model_name,
        device,
        resolved_precision,
        attention,
    )
    return model, resolved_precision, attention


def _token_lengths(model, texts: tuple[str, ...]) -> list[int]:
    if not texts:
        return []
    max_length = getattr(model, "max_seq_length", None)
    encoded = model.tokenizer(
        list(texts),
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    return [len(ids) for ids in encoded["input_ids"]]


def _batch_cost(lengths: list[int], *, unpadded: bool) -> int:
    if not lengths:
        return 0
    if unpadded:
        return sum(lengths)
    return len(lengths) * max(lengths)


def _target_token_span(
    offset_mapping: list[tuple[int, int]],
    char_offset: tuple[int, int],
    *,
    item_index: int,
) -> tuple[int, int]:
    """Map a character span to token indices using Graphwise's legacy semantics.

    Tokenizer offset mappings intentionally do not cover whitespace and special tokens.
    Therefore a requested character span does not need to be fully covered by token
    offsets: the first token whose end lies after ``char_start`` begins the span, and
    the first token whose start is at/after ``char_end`` ends it.  This preserves the
    original Graphwise API behaviour (including spans that begin/end in whitespace)
    and keeps the golden embeddings backward compatible.
    """
    char_start, char_end = char_offset
    if char_start < 0 or char_end <= char_start:
        raise ValueError(
            f"Invalid character offset for input {item_index}: {char_offset}"
        )

    start_token = end_token = None
    for token_idx, (token_start, token_end) in enumerate(offset_mapping):
        if start_token is None and token_end > char_start:
            start_token = token_idx

        if end_token is None and token_start >= char_end:
            end_token = token_idx

        # Once the span has started, a zero-width special token (normally SEP)
        # terminates it if no ordinary token boundary did so first.
        if (
            end_token is None
            and start_token is not None
            and token_start == token_end == 0
        ):
            end_token = token_idx

        if start_token is not None and end_token is not None:
            break

    if start_token is None:
        start_token = 0
    if end_token is None or end_token <= start_token:
        end_token = min(start_token + 1, len(offset_mapping))

    return start_token, end_token


def _input_ids_list(value) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [int(x) for x in value]


def _output_kind(item: _WorkItem) -> str:
    embedding_type = EmbeddingType(item.embedding_type)
    return "sentence" if embedding_type == EmbeddingType.SENTENCE else "token"


def _encode_token_outputs(model, texts: list[str], batch_size: int) -> list[_TokenOutput]:
    """Encode token vectors through SentenceTransformers' supported public API.

    SentenceTransformers 5.6.1 supports FA2 input unpadding for sentence-level
    encoding, but its public ``output_value="token_embeddings"`` post-processing
    expects an ordinary padded batch. Use the documented ``unpad_inputs`` switch
    to force padding only for this token-level encode, then restore the model's
    previous setting. FlashAttention itself remains the selected attention backend.
    """
    transformer_module = model[0]
    has_unpad_control = hasattr(transformer_module, "unpad_inputs")
    previous_unpad = transformer_module.unpad_inputs if has_unpad_control else None

    if has_unpad_control:
        transformer_module.unpad_inputs = False
    try:
        token_embeddings = model.encode(
            texts,
            batch_size=max(1, batch_size),
            output_value="token_embeddings",
            convert_to_numpy=False,
            convert_to_tensor=False,
            show_progress_bar=False,
        )
    finally:
        if has_unpad_control:
            transformer_module.unpad_inputs = previous_unpad

    max_length = getattr(model, "max_seq_length", None)
    encoded = model.tokenizer(
        texts,
        padding=False,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"]

    if len(token_embeddings) != len(input_ids):
        raise RuntimeError(
            "SentenceTransformer returned an unexpected number of token outputs: "
            f"{len(token_embeddings)} for {len(input_ids)} tokenized inputs"
        )

    outputs: list[_TokenOutput] = []
    for index, (embeddings, ids) in enumerate(zip(token_embeddings, input_ids)):
        ids = _input_ids_list(ids)
        if len(embeddings) != len(ids):
            raise RuntimeError(
                "Token embedding/token-id length mismatch for input "
                f"{index}: {len(embeddings)} != {len(ids)}"
            )
        outputs.append(_TokenOutput(token_embeddings=embeddings, input_ids=ids))
    return outputs


def _postprocess_sentence(item: _WorkItem, embeddings) -> list[list[Embedding]]:
    return [
        [
            Embedding(
                string=item.texts[i],
                embedding=embedding.detach().float().cpu().tolist(),
            )
        ]
        for i, embedding in enumerate(embeddings)
    ]


def _postprocess_token_task(
    model, item: _WorkItem, outputs: list[_TokenOutput]
) -> list[list[Embedding]]:
    embedding_type = EmbeddingType(item.embedding_type)
    tokenizer = model.tokenizer

    if embedding_type == EmbeddingType.TOKEN:
        results: list[list[Embedding]] = []
        for output in outputs:
            token_embeddings = output.token_embeddings.detach().float().cpu()
            tokens = tokenizer.convert_ids_to_tokens(output.input_ids)
            if len(token_embeddings) != len(tokens):
                raise RuntimeError(
                    "Token embedding/token-id length mismatch after unpacking: "
                    f"{len(token_embeddings)} != {len(tokens)}"
                )
            results.append(
                [
                    Embedding(string=token, embedding=embedding.tolist())
                    for embedding, token in zip(token_embeddings, tokens)
                ]
            )
        return results

    if embedding_type != EmbeddingType.OFFSET:
        raise ValueError(f"Expected token or offset embedding type, got {embedding_type}")

    if item.char_offsets is None or len(item.char_offsets) != len(item.texts):
        raise ValueError(
            "Offsets must be provided and must have the same length as the input "
            "in EmbeddingType.OFFSET"
        )

    max_length = getattr(model, "max_seq_length", None)
    tokenized = tokenizer(
        list(item.texts),
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        padding=False,
    )

    results: list[list[Embedding]] = []
    for i, (output, char_offset, offset_mapping) in enumerate(
        zip(outputs, item.char_offsets, tokenized["offset_mapping"])
    ):
        if len(output.input_ids) != len(offset_mapping):
            raise RuntimeError(
                "Token embedding/offset length mismatch after unpadding for input "
                f"{i}: {len(output.input_ids)} != {len(offset_mapping)}"
            )
        start_token, end_token = _target_token_span(
            offset_mapping, char_offset, item_index=i
        )
        token_embeddings = output.token_embeddings[start_token:end_token]
        if len(token_embeddings) == 0:
            raise ValueError(f"Offset produced an empty token span for input {i}")
        embedding = token_embeddings.detach().float().cpu().numpy().mean(axis=0)
        text = tokenizer.decode(
            output.input_ids[start_token:end_token],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        results.append([Embedding(string=text, embedding=embedding.tolist())])
    return results


def _worker_main(
    model_name: str,
    device: str,
    precision: str,
    flash_attention: bool,
    max_batch_tokens: int,
    max_batch_wait_ms: float,
    input_queue,
    result_queue,
    ready_queue,
) -> None:
    try:
        model, resolved_precision, attention = _load_sentence_transformer(
            model_name, device, precision, flash_attention
        )
        ready_queue.put(
            _WorkerReady(
                device=device,
                precision=resolved_precision,
                attention=attention,
            )
        )
    except Exception as exc:
        ready_queue.put(
            _WorkerReady(
                device=device,
                precision="unknown",
                attention="unknown",
                error=f"{type(exc).__name__}: {exc}",
            )
        )
        logging.exception("Worker failed to initialize model %s on %s", model_name, device)
        return

    # Sentence-level FA2 requests use SentenceTransformers' documented automatic
    # input unpadding. TOKEN/OFFSET use the documented padded token-output path.
    carry: _WorkItem | None = None
    shutdown_after_batch = False

    while True:
        if carry is not None:
            first = carry
            carry = None
        else:
            first = input_queue.get()
        if first is None:
            break

        batch_unpadded = (
            attention == "flash_attention_2" and _output_kind(first) == "sentence"
        )

        try:
            first_lengths = _token_lengths(model, first.texts)
        except Exception as exc:
            result_queue.put(
                _WorkResult(
                    request_id=first.request_id,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            continue

        batch: list[tuple[_WorkItem, list[int]]] = [(first, first_lengths)]
        all_lengths = list(first_lengths)
        input_count = len(first.texts)
        deadline = time.monotonic() + max_batch_wait_ms / 1000.0

        while input_count < _MAX_BATCH_INPUTS:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                nxt = input_queue.get(timeout=remaining)
            except queue.Empty:
                break
            if nxt is None:
                shutdown_after_batch = True
                break

            # A SentenceTransformer encode call has one output contract. Sentence
            # embeddings and token-level outputs therefore use separate microbatches.
            # TOKEN and OFFSET intentionally share the token-level contract.
            if _output_kind(nxt) != _output_kind(first):
                carry = nxt
                break

            try:
                nxt_lengths = _token_lengths(model, nxt.texts)
            except Exception as exc:
                result_queue.put(
                    _WorkResult(
                        request_id=nxt.request_id,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                continue

            prospective_lengths = all_lengths + nxt_lengths
            prospective_inputs = input_count + len(nxt.texts)
            if (
                batch
                and (
                    prospective_inputs > _MAX_BATCH_INPUTS
                    or _batch_cost(prospective_lengths, unpadded=batch_unpadded)
                    > max_batch_tokens
                )
            ):
                carry = nxt
                break

            batch.append((nxt, nxt_lengths))
            all_lengths = prospective_lengths
            input_count = prospective_inputs

        flat_texts: list[str] = []
        slices: list[tuple[_WorkItem, int, int]] = []
        for item, _lengths in batch:
            start = len(flat_texts)
            flat_texts.extend(item.texts)
            slices.append((item, start, len(flat_texts)))

        if not flat_texts:
            for item, _start, _end in slices:
                result_queue.put(_WorkResult(request_id=item.request_id, results=[]))
            if shutdown_after_batch:
                break
            continue

        # If one RPC itself exceeds the token budget, split inference into
        # conservative sub-batches. Sentence requests may use FA2 unpadding;
        # token-level requests are intentionally budgeted as padded batches.
        max_len = max(all_lengths) if all_lengths else 1
        if _batch_cost(all_lengths, unpadded=batch_unpadded) <= max_batch_tokens:
            encode_batch_size = len(flat_texts)
        else:
            encode_batch_size = max(1, min(len(flat_texts), max_batch_tokens // max_len))

        try:
            output_kind = _output_kind(batch[0][0])
            if output_kind == "sentence":
                outputs = model.encode(
                    flat_texts,
                    batch_size=encode_batch_size,
                    output_value="sentence_embedding",
                    convert_to_numpy=False,
                    convert_to_tensor=True,
                    normalize_embeddings=False,
                    show_progress_bar=False,
                )
            else:
                outputs = _encode_token_outputs(
                    model, flat_texts, batch_size=encode_batch_size
                )

            if len(outputs) != len(flat_texts):
                raise RuntimeError(
                    "SentenceTransformer returned an unexpected number of outputs: "
                    f"{len(outputs)} for {len(flat_texts)} inputs"
                )
        except Exception as exc:
            logging.error(
                "Inference failed for %s on %s: %s\n%s",
                model_name,
                device,
                exc,
                traceback.format_exc(),
            )
            for item, _start, _end in slices:
                result_queue.put(
                    _WorkResult(
                        request_id=item.request_id,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
            if shutdown_after_batch:
                break
            continue

        for item, start, end in slices:
            try:
                item_outputs = outputs[start:end]
                if _output_kind(item) == "sentence":
                    results = _postprocess_sentence(item, item_outputs)
                else:
                    results = _postprocess_token_task(model, item, item_outputs)
                result_queue.put(
                    _WorkResult(
                        request_id=item.request_id,
                        results=results,
                    )
                )
            except Exception as exc:
                result_queue.put(
                    _WorkResult(
                        request_id=item.request_id,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )

        if shutdown_after_batch:
            break


def _visible_worker_devices(
    cpu_workers: int, runtime_target: str = "auto"
) -> list[str]:
    if runtime_target == "cpu":
        if cpu_workers <= 0:
            raise RuntimeError("CPU runtime requires at least one CPU worker")
        return ["cpu"] * cpu_workers

    if runtime_target == "gpu":
        if cpu_workers != 0:
            raise RuntimeError("GPU runtime does not permit CPU workers")
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("GPU runtime requires at least one visible CUDA device")
        count = torch.cuda.device_count()
        if count <= 0:
            raise RuntimeError("GPU runtime requires at least one visible CUDA device")
        return [f"cuda:{index}" for index in range(count)]

    if torch is not None and torch.cuda.is_available():
        count = torch.cuda.device_count()
        if count > 0:
            return [f"cuda:{index}" for index in range(count)]

    if cpu_workers <= 0:
        raise RuntimeError("No CUDA devices are visible and CPU workers are disabled")
    return ["cpu"] * cpu_workers


class EmbeddingModel:
    """Process-isolated SentenceTransformer serving pool for one logical model."""

    def __init__(
        self,
        model_name: str | None = None,
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
        self._model_name = model_name or DEFAULT_MODEL
        self._request_timeout_seconds = request_timeout_seconds
        self._ctx = mp.get_context("spawn")
        self._input_queue = self._ctx.Queue(maxsize=max_queue_size)
        self._result_queue = self._ctx.Queue()
        self._ready_queue = self._ctx.Queue()
        self._workers: list[mp.Process] = []
        self._pending: dict[str, queue.Queue[_WorkResult]] = {}
        self._pending_lock = threading.Lock()
        self._closed = False

        devices = _visible_worker_devices(cpu_workers, runtime_target)
        logging.info(
            "Starting %d worker process(es) for %s on %s",
            len(devices),
            self._model_name,
            ", ".join(devices),
        )
        for device in devices:
            process = self._ctx.Process(
                target=_worker_main,
                args=(
                    self._model_name,
                    device,
                    precision,
                    flash_attention,
                    max_batch_tokens,
                    max_batch_wait_ms,
                    self._input_queue,
                    self._result_queue,
                    self._ready_queue,
                ),
                name=f"graphwise-{device.replace(':', '-')}",
                daemon=True,
            )
            process.start()
            self._workers.append(process)

        ready: list[_WorkerReady] = []
        deadline = time.monotonic() + worker_start_timeout_seconds
        while len(ready) < len(self._workers):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.unload()
                raise TimeoutError(
                    f"Timed out loading {self._model_name} worker processes"
                )
            try:
                ready.append(self._ready_queue.get(timeout=remaining))
            except queue.Empty as exc:
                self.unload()
                raise TimeoutError(
                    f"Timed out loading {self._model_name} worker processes"
                ) from exc

        errors = [status for status in ready if status.error]
        if errors:
            self.unload()
            details = "; ".join(f"{x.device}: {x.error}" for x in errors)
            raise RuntimeError(f"Failed to load {self._model_name}: {details}")

        self._collector = threading.Thread(
            target=self._collect_results,
            name=f"graphwise-results-{self._model_name}",
            daemon=True,
        )
        self._collector.start()

    @property
    def worker_count(self) -> int:
        return len(self._workers)

    def _collect_results(self) -> None:
        while True:
            result = self._result_queue.get()
            if result is None:
                return
            with self._pending_lock:
                waiter = self._pending.get(result.request_id)
            if waiter is not None:
                waiter.put(result)

    def _raise_worker_error(self, result: _WorkResult) -> None:
        message = result.error_message or "Worker inference failed"
        if result.error_type == "ValueError":
            raise ValueError(message)
        if result.error_type == "KeyError":
            raise KeyError(message)
        raise RuntimeError(f"{result.error_type or 'RuntimeError'}: {message}")

    def embed(
        self,
        embedding_type: EmbeddingType = EmbeddingType.OFFSET,
        texts: list[str] | None = None,
        char_offsets: list[tuple[int, int]] | None = None,
    ) -> list[list[Embedding]]:
        if self._closed:
            raise RuntimeError("Model is not loaded")
        if texts is None:
            raise ValueError("texts must be provided")
        if embedding_type == EmbeddingType.OFFSET and (
            char_offsets is None or len(char_offsets) != len(texts)
        ):
            raise ValueError(
                "Offsets must be provided and must have the same length as the input "
                "in EmbeddingType.OFFSET"
            )
        if not texts:
            return []

        request_id = uuid.uuid4().hex
        waiter: queue.Queue[_WorkResult] = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[request_id] = waiter

        item = _WorkItem(
            request_id=request_id,
            embedding_type=embedding_type.value,
            texts=tuple(texts),
            char_offsets=tuple(char_offsets) if char_offsets is not None else None,
        )
        try:
            self._input_queue.put(item, timeout=self._request_timeout_seconds)
            result = waiter.get(timeout=self._request_timeout_seconds)
        except queue.Full as exc:
            raise RuntimeError("Inference queue is full") from exc
        except queue.Empty as exc:
            dead = [worker.name for worker in self._workers if not worker.is_alive()]
            detail = f"; dead workers: {dead}" if dead else ""
            raise TimeoutError(
                f"Timed out waiting for inference result{detail}"
            ) from exc
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)

        if result.error_type:
            self._raise_worker_error(result)
        return result.results or []

    def unload(self) -> None:
        if self._closed:
            return
        self._closed = True

        for _ in self._workers:
            try:
                self._input_queue.put_nowait(None)
            except queue.Full:
                # Workers are draining the queue; blocking briefly is safer than
                # abandoning a process during graceful unload.
                try:
                    self._input_queue.put(None, timeout=1.0)
                except queue.Full:
                    break

        for worker in self._workers:
            worker.join(timeout=10.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5.0)

        with self._pending_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for waiter in pending:
            try:
                waiter.put_nowait(
                    _WorkResult(
                        request_id="",
                        error_type="RuntimeError",
                        error_message="Model was unloaded",
                    )
                )
            except queue.Full:
                pass

        try:
            self._result_queue.put_nowait(None)
        except Exception:
            pass
        collector = getattr(self, "_collector", None)
        if collector is not None:
            collector.join(timeout=2.0)

        for q in (self._input_queue, self._result_queue, self._ready_queue):
            try:
                q.close()
            except Exception:
                pass
