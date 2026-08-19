# graphwise-transformer

A Python gRPC server for SentenceTransformer embedding models. It exposes sentence, token, and character-offset embeddings through the Graphwise Transformer protobuf API and supports loading and unloading models at runtime.

The server is designed for concurrent inference without sharing a `SentenceTransformer` instance between request threads. Each visible GPU receives an independent worker process and model instance; when no GPU is visible, the configured number of CPU workers is used. Requests are exchanged through process-safe queues and are dynamically combined into microbatches inside the workers.

## Architecture

```text
GraphDB / gRPC client
        |
        | transformer.proto, port 5050
        v
graphwise-transformer
        |
        | bounded multiprocessing queue
        v
+-------------------------------+
| SentenceTransformer workers   |
|                               |
| GPU 0 -> process + model      |
| GPU 1 -> process + model      |
| ...                           |
|                               |
| no GPU -> CPU worker(s)       |
+-------------------------------+
        |
        +-> dynamic microbatching
        +-> FlashAttention-2 + input unpadding when supported
        +-> PyTorch SDPA fallback otherwise
```

A worker owns its tokenizer, model, CUDA context, and inference state for its entire lifetime. No model or tokenizer object is shared between workers, and inference does not use a global model lock.

For every loaded logical model, the registry starts one worker process per CUDA device visible inside the container. For example, a container that can see three GPUs starts three independent replicas of that model. All replicas consume from the same request queue, so work is naturally distributed among available devices.

Workers collect queued requests for a short configurable window and combine compatible requests into microbatches. `SENTENCE` requests are batched together and use `SentenceTransformer.encode(..., output_value="sentence_embedding")`; `TOKEN` and `OFFSET` requests share a token-level microbatch path using the public `output_value="token_embeddings"` API. Sentence-level FA2 batches use the real-token budget because SentenceTransformers can unpad them. Token-level batches are conservatively budgeted as padded batches because the documented token-output path is run with unpadding disabled.

## Embedding modes

The protobuf contract is unchanged and is defined in `protos/transformer.proto`.

### InferenceService

- `EmbedSentence(SentenceRequest) -> SentenceResponse`
- `EmbedTokens(TokenRequest) -> TokenResponse`
- `EmbedWithOffsets(OffsetRequest) -> OffsetResponse`

### AdminService

- `LoadModel(LoadModelRequest) -> LoadModelResponse`
- `UnloadModel(UnloadModelRequest) -> UnloadModelResponse`
- `ListModels(ListModelsRequest) -> ListModelsResponse`

`SENTENCE` uses SentenceTransformers' public `output_value="sentence_embedding"` API. `TOKEN` and `OFFSET` use its public `output_value="token_embeddings"` API. The server does not use `encode(output_value=None)` and does not inspect or reconstruct SentenceTransformers' private packed-feature representation. Offset metadata is obtained separately from the model tokenizer with the same truncation limit; offset embeddings are mean pooled over the token span overlapping the requested character range. An offset that is outside the tokenizer-visible/truncated portion of the input is rejected rather than silently returning an unrelated embedding.

## Runtime configuration

The default `config.properties` is:

```properties
port=5050
log_level=INFO
default_model=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
max_workers=8

precision=auto
flash_attention=true

max_queue_size=1024
max_batch_tokens=16384
max_batch_wait_ms=2

cpu_workers=1
worker_start_timeout_seconds=300
request_timeout_seconds=300
```

The configuration file can be selected with:

```bash
GRAPHWISE_CONFIG=/path/to/config.properties
```

The serving settings can also be overridden individually:

| Property | Environment variable | Description |
|---|---|---|
| `precision` | `GRAPHWISE_PRECISION` | `auto`, `fp16`, `bf16`, or `fp32` |
| `flash_attention` | `GRAPHWISE_FLASH_ATTENTION` | Prefer FlashAttention-2 when installed and supported |
| `max_queue_size` | `GRAPHWISE_MAX_QUEUE_SIZE` | Maximum queued RPC work items per loaded model |
| `max_batch_tokens` | `GRAPHWISE_MAX_BATCH_TOKENS` | Token budget used when workers combine requests |
| `max_batch_wait_ms` | `GRAPHWISE_MAX_BATCH_WAIT_MS` | Maximum microbatch collection delay |
| `cpu_workers` | `GRAPHWISE_CPU_WORKERS` | CPU target process count; default `1`. GPU target resolves it to `0` and rejects non-zero values |
| `worker_start_timeout_seconds` | `GRAPHWISE_WORKER_START_TIMEOUT_SECONDS` | Maximum model-worker startup time |
| `request_timeout_seconds` | `GRAPHWISE_REQUEST_TIMEOUT_SECONDS` | Queue/result timeout for an inference RPC |

GPU worker count is intentionally not a configuration option. It is derived from the CUDA devices visible to the container.

### Precision

`precision=auto` uses:

- CUDA with native BF16 support: BF16
- other CUDA devices: FP16
- CPU: FP32

The returned protobuf embedding values remain ordinary floating-point values; the setting controls model compute precision, not output quantization.

## FlashAttention and input unpadding

The pinned runtime stack is:

- PyTorch `2.9.1`
- CUDA `12.8` PyTorch wheels in the GPU image
- Transformers `5.14.1`
- SentenceTransformers `5.6.1`
- FlashAttention `2.8.3` in the default GPU image

Transformers 5.x is intentional. SentenceTransformers automatic variable-length input unpadding for FlashAttention requires Transformers >= 5.0.0. For sentence embeddings, workers keep the documented `model[0].unpad_inputs = None` automatic mode so compatible text-only models can use the standard variable-length FA2 path.

For `TOKEN` and `OFFSET`, the worker temporarily sets the documented `model[0].unpad_inputs = False` control and calls `SentenceTransformer.encode(..., output_value="token_embeddings")`. SentenceTransformers 5.6.1's public token-output post-processing expects a padded batch, so this is the supported path rather than depending on SentenceTransformers' internal packed-feature representation. The previous `unpad_inputs` value is restored in a `finally` block after every token-level encode. FlashAttention-2 remains the attention implementation; only the input-flattening optimization is disabled for token-level RPCs.

Stable FlashAttention-2 officially targets Ampere, Ada, and Hopper NVIDIA GPUs. On unsupported hardware, when FlashAttention is not installed, or when `GRAPHWISE_FLASH_ATTENTION=false`, workers use PyTorch SDPA.

The GPU image does **not** compile FlashAttention. `requirements_gpu.in` pins the official upstream FlashAttention `2.8.3` binary wheel for Linux x86_64, CPython 3.12, Torch 2.9.x, CUDA 12.x, and the CXX11 ABI, including its SHA-256. The GPU requirements also pin `torch==2.9.1+cu128`, so the binary stack is selected entirely by the GPU requirements file.

No CUDA toolkit, `nvcc`, compiler, Ninja, or FlashAttention source build is required. CUDA user-space libraries come from the official PyTorch cu128 dependency stack; the NVIDIA driver is supplied by the host through the NVIDIA Container Toolkit.

FlashAttention is installed in the GPU target. Runtime use can still be disabled without rebuilding:

```bash
GRAPHWISE_FLASH_ATTENTION=false
```

## Docker

The Dockerfile has two runtime targets only:

- `cpu` — CPU-only PyTorch, no CUDA packages and no FlashAttention.
- `gpu` — PyTorch cu128 plus the pinned official FlashAttention wheel.

Both runtime targets are based on `python:3.12-slim-bookworm`.

### CPU image

Build:

```bash
docker build \
  --target cpu \
  -t graphwise-transformer:cpu .
```

Run:

```bash
docker run --rm \
  -p 5050:5050 \
  graphwise-transformer:cpu
```

No NVIDIA runtime or GPU is required. The CPU target defaults to exactly one SentenceTransformer worker (`GRAPHWISE_CPU_WORKERS=1`) and resolves `precision=auto` to FP32. Set `GRAPHWISE_CPU_WORKERS` to a larger positive value when multiple CPU model processes are desired.

Docker Compose:

```yaml
services:
  graphwisetransformer:
    build:
      context: .
      target: cpu
    ports:
      - "5050:5050"
    environment:
      GRAPHWISE_PRECISION: fp32
      GRAPHWISE_CPU_WORKERS: "4"
      GRAPHWISE_MAX_QUEUE_SIZE: "1024"
      GRAPHWISE_MAX_BATCH_TOKENS: "16384"
      GRAPHWISE_MAX_BATCH_WAIT_MS: "2"
      GRAPHWISE_WORKER_START_TIMEOUT_SECONDS: "300"
      GRAPHWISE_REQUEST_TIMEOUT_SECONDS: "300"
    volumes:
      - hf-cache:/home/appuser/.cache/huggingface

volumes:
  hf-cache:
```

`GRAPHWISE_CPU_WORKERS` controls the CPU target's model-process count and must be greater than zero. The CPU target is authoritative: even if a GPU is accidentally exposed to the container, it still creates CPU workers only. The CPU image never installs CUDA PyTorch or FlashAttention. `precision=auto` resolves to FP32 on CPU, so setting `GRAPHWISE_PRECISION=fp32` is explicit but not required.

### GPU image

Build the default GPU image with FlashAttention:

```bash
docker build \
  --target gpu \
  -t graphwise-transformer:gpu .
```

The host needs a compatible NVIDIA driver and NVIDIA Container Toolkit. It does **not** need a system CUDA toolkit because the final image uses the CUDA runtime packages installed with the cu128 PyTorch wheel.

#### One GPU

Select a single physical GPU at the container boundary:

```bash
docker run --rm \
  --gpus '"device=1"' \
  -p 5050:5050 \
  graphwise-transformer:gpu
```

Exactly one GPU is visible inside the container, so the server creates one GPU worker/model replica.

Docker Compose:

```yaml
services:
  graphwisetransformer:
    build:
      context: .
      target: gpu
    ports:
      - "5050:5050"
    environment:
      GRAPHWISE_PRECISION: auto
      GRAPHWISE_FLASH_ATTENTION: "true"
      GRAPHWISE_MAX_QUEUE_SIZE: "1024"
      GRAPHWISE_MAX_BATCH_TOKENS: "16384"
      GRAPHWISE_MAX_BATCH_WAIT_MS: "2"
      GRAPHWISE_WORKER_START_TIMEOUT_SECONDS: "300"
      GRAPHWISE_REQUEST_TIMEOUT_SECONDS: "300"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]
              capabilities: [gpu]
    volumes:
      - hf-cache:/home/appuser/.cache/huggingface

volumes:
  hf-cache:
```

`GRAPHWISE_PRECISION=auto` selects BF16 on CUDA devices that report native BF16 support and FP16 otherwise. Set it explicitly to `bf16`, `fp16`, or `fp32` when reproducibility across different GPU types is more important than automatic selection. `GRAPHWISE_FLASH_ATTENTION` controls runtime use of the FlashAttention package already installed in the GPU image.

The GPU target resolves `GRAPHWISE_CPU_WORKERS` to `0` by default. Do not override it with a non-zero value: configuration validation rejects CPU workers in a GPU image. If no CUDA device is visible, the GPU target does not fall back to CPU inference.

A GPU UUID can be used instead of an ordinal when stable device identity is preferred.

#### Multiple GPUs

Expose all GPUs that should participate in inference to the same container:

```yaml
services:
  graphwisetransformer:
    build:
      context: .
      target: gpu
    ports:
      - "5050:5050"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0", "2", "3"]
              capabilities: [gpu]
    volumes:
      - hf-cache:/home/appuser/.cache/huggingface

volumes:
  hf-cache:
```

Graphwise Transformer sees only the assigned devices and starts one independent SentenceTransformer worker process per visible CUDA device while exposing one gRPC endpoint:

```text
graphwisetransformer:5050
        |
        +-> worker cuda:0 -> model replica
        +-> worker cuda:1 -> model replica
        +-> worker cuda:2 -> model replica
```

The CUDA indices above are container-local indices. Each worker owns its tokenizer, CUDA context, and complete model replica. The model is data-parallel across workers; it is not tensor-sharded across GPUs.

Requests from concurrent clients enter the same bounded model queue. Available workers take queued work, opportunistically merge compatible requests into token-budgeted microbatches, and return results by request ID. This provides one public gRPC endpoint without sharing a model instance or tokenizer across processes.

## Docker Compose configuration

The Docker target selects both the dependency stack and the permitted worker type. Use `build.target: cpu` for CPU inference and `build.target: gpu` for CUDA inference. The CPU target defaults to one CPU worker; the GPU target sets CPU workers to zero and rejects non-zero CPU-worker configuration. Do not expose NVIDIA devices to the CPU target.

The properties `port`, `log_level`, `default_model`, `max_workers`, and optional `secret` are read from `config.properties`. To provide a deployment-specific file through Compose, mount it read-only and point `GRAPHWISE_CONFIG` at it:

```yaml
services:
  graphwisetransformer:
    build:
      context: .
      target: gpu
    environment:
      GRAPHWISE_CONFIG: /etc/graphwise/config.properties
      # Serving settings below override the same properties in the file.
      GRAPHWISE_PRECISION: bf16
      GRAPHWISE_FLASH_ATTENTION: "true"
      GRAPHWISE_MAX_QUEUE_SIZE: "2048"
      GRAPHWISE_MAX_BATCH_TOKENS: "32768"
      GRAPHWISE_MAX_BATCH_WAIT_MS: "2"
      GRAPHWISE_WORKER_START_TIMEOUT_SECONDS: "300"
      GRAPHWISE_REQUEST_TIMEOUT_SECONDS: "300"
    volumes:
      - ./config.properties:/etc/graphwise/config.properties:ro
      - hf-cache:/home/appuser/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]

volumes:
  hf-cache:
```

The supported per-setting environment overrides are:

| Environment variable | Applies to | Typical use |
|---|---|---|
| `GRAPHWISE_CONFIG` | CPU/GPU | Path to `config.properties` |
| `GRAPHWISE_PRECISION` | CPU/GPU | `auto`, `fp32`, `fp16`, or `bf16` |
| `GRAPHWISE_FLASH_ATTENTION` | GPU | Enable FA2 when installed/supported; otherwise SDPA |
| `GRAPHWISE_MAX_QUEUE_SIZE` | CPU/GPU | Bound queued inference work per loaded model |
| `GRAPHWISE_MAX_BATCH_TOKENS` | CPU/GPU | Maximum token budget for a worker microbatch |
| `GRAPHWISE_MAX_BATCH_WAIT_MS` | CPU/GPU | Maximum time to wait while coalescing a microbatch |
| `GRAPHWISE_CPU_WORKERS` | CPU only | Number of CPU SentenceTransformer processes; default `1`. GPU target requires `0` |
| `GRAPHWISE_WORKER_START_TIMEOUT_SECONDS` | CPU/GPU | Model-worker startup timeout |
| `GRAPHWISE_REQUEST_TIMEOUT_SECONDS` | CPU/GPU | Per-RPC inference result timeout |

`max_workers` is the gRPC server thread-pool size. It is not the number of model processes. In the GPU target, process count is always the number of CUDA devices visible inside the container and CPU workers are forbidden. In the CPU target, process count comes from `cpu_workers` and defaults to one.

## Model cache

The image sets `HF_HOME=/home/appuser/.cache/huggingface` and creates that directory as the non-root runtime user (UID `10001`). For Docker deployments, mount the Hugging Face cache if models should persist across container recreations:

```yaml
services:
  graphwisetransformer:
    volumes:
      - hf-cache:/home/appuser/.cache/huggingface

volumes:
  hf-cache:
```

A newly created `hf-cache` volume inherits the cache directory prepared by the image and is writable by `appuser`. If the volume already existed from an older image and is root-owned, repair it once:

```bash
docker compose run --rm --user root --entrypoint sh graphwisetransformer \
  -c 'mkdir -p /home/appuser/.cache/huggingface && chown -R 10001:10001 /home/appuser/.cache/huggingface'
```

Then start the service normally. Rebuilding the image alone cannot change ownership stored inside an existing Docker volume.

## Local development

The Docker images are the reference environments. For local CPU development:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_cpu.txt
python setup.py build
pytest -q
```

Run the server with:

```bash
python -m graphwise_transformer.server
```

The golden model integration suite is intentionally opt-in because it downloads/loads real Hugging Face models:

```bash
RUN_MODEL_INTEGRATION=1 pytest -q tests/test_integration_model.py
```

## Dependency files

Dependencies are split by Docker target:

- `requirements_cpu.in` pins the CPU stack, including `torch==2.9.1+cpu`.
- `requirements_gpu.in` pins the CUDA 12.8 stack, including `torch==2.9.1+cu128` and the hash-pinned official FlashAttention wheel.
- `requirements_cpu.txt` and `requirements_gpu.txt` are the files consumed by the Docker targets. Recompile them from the corresponding `.in` files whenever dependencies change.

Using `pip-tools`:

```bash
pip-compile requirements_cpu.in -o requirements_cpu.txt
pip-compile requirements_gpu.in -o requirements_gpu.txt
```

The target-specific Torch local-version pins (`+cpu` and `+cu128`) make the desired PyTorch build explicit even though the PyTorch repositories are configured as extra indexes in the `.in` files.

## Versioning

The project version is defined in `pyproject.toml` and exposed as `graphwise_transformer.__version__`.

## Author

- **Tomas Kovachev** - tomas.kovachev@graphwise.ai
