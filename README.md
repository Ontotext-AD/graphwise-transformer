# graphwise-transformer

A Python gRPC server that serves SentenceTransformer models for generating embeddings. It provides:
- Inference APIs for sentence, token, and offset-span embeddings
- Admin APIs to load/unload models at runtime
- Configurable via `config.properties`

## Architecture
- gRPC only (no HTTP). Protobufs under `protos/transformer.proto` with Python stubs generated at build time
- Core components:
  - `graphwise_transformer/model.py`: wraps SentenceTransformer and implements three embedding modes (SENTENCE, TOKEN, OFFSET)
  - `graphwise_transformer/registry.py`: thread-safe registry for loading/unloading models
  - `graphwise_transformer/server.py`: gRPC services using generated stubs
  - `graphwise_transformer/config.py`: simple properties loader (`GRAPHWISE_CONFIG` env var overrides path)
- Build-time proto generation wired in `setup.py` (runs automatically on `build`)

## Configuration
`config.properties` (defaults provided):
```
port=5050
log_level=INFO
default_model=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
max_workers=8
```
Override with env var `GRAPHWISE_CONFIG=/path/to/config.properties`.

## gRPC API (summary)
- Package: `graphwise_transformer`
- InferenceService
  - `EmbedSentence(SentenceRequest) -> SentenceResponse`
  - `EmbedTokens(TokenRequest) -> TokenResponse`
  - `EmbedWithOffsets(OffsetRequest) -> OffsetResponse`
- AdminService
  - `LoadModel(LoadModelRequest) -> LoadModelResponse`
  - `UnloadModel(UnloadModelRequest) -> UnloadModelResponse`
  - `ListModels(ListModelsRequest) -> ListModelsResponse`

Key messages (see proto for full details):
- `SentenceRequest { string model_name; repeated string texts; }`
- `TokenRequest { string model_name; repeated string texts; }`
- `TextWithOffsets { string text; int32 start; int32 end; }`
- `OffsetRequest { string model_name; repeated TextWithOffsets inputs; }`
- `Embedding { string string; repeated float embedding; }`
- `TokenEmbeddings { repeated Embedding tokens; }`

## Local development

### Prerequisites
- Python 3.10+
- `pip`

### Install dependencies
```
pip install -r requirements.txt
```

### Generate stubs and build
```
python setup.py build
```
This will generate Python gRPC stubs into `graphwise_transformer/proto`.

### Run the server
```
python -m graphwise_transformer.server
```
The server listens on the configured `port`.

### Run tests
```
pytest -q
```

## Docker
An image is provided. It uses a slim Python base, installs dependencies, generates gRPC stubs at build time, and runs the server as a non-root user.

### Build image
```
docker build -t graphwise-transformer:$(git rev-parse --short HEAD) .
```

### Run container
```
docker run --rm -p 5050:5050 \
  -e GRAPHWISE_CONFIG=/app/config.properties \
  graphwise-transformer:$(git rev-parse --short HEAD)
```
Mount or bake your own `config.properties` if you need different settings; the default inside the image is suitable for local runs.

## Versioning
- Project version is defined in `pyproject.toml` and exposed as `graphwise_transformer.__version__`.

## Author
- **Tomas Kovachev** - tomas.kovachev@graphwise.ai

## Notes
- GPU: If CUDA is available, unloading a model clears the CUDA cache to release VRAM.
- Offsets: Offset requests expect one `(start,end)` span per input text.
