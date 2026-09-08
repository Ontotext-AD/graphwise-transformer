import configparser
import logging
import os
from dataclasses import dataclass


VALID_PRECISIONS = {"auto", "fp16", "bf16", "fp32"}
VALID_RUNTIME_TARGETS = {"auto", "cpu", "gpu"}


@dataclass(frozen=True)
class AppConfig:
    port: int
    log_level: str
    default_model: str
    max_workers: int
    secret: str | None
    precision: str = "auto"
    flash_attention: bool = True
    max_queue_size: int = 1024
    max_batch_tokens: int = 16384
    max_batch_wait_ms: float = 2.0
    cpu_workers: int = 1
    runtime_target: str = "auto"
    worker_start_timeout_seconds: float = 300.0
    request_timeout_seconds: float = 300.0


def _validate_choice(value: str, valid: set[str], name: str) -> str:
    normalized = value.strip().lower()
    if normalized not in valid:
        raise ValueError(
            f"Invalid {name} '{value}'; expected one of {sorted(valid)}"
        )
    return normalized


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value '{value}'")


def _positive_int(value: str | int, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0, got {parsed}")
    return parsed


def _nonnegative_int(value: str | int, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be >= 0, got {parsed}")
    return parsed


def _positive_float(value: str | float, name: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0, got {parsed}")
    return parsed


def load_config(config_path: str | None = None) -> AppConfig:
    if config_path is None:
        config_path = os.environ.get(
            "GRAPHWISE_CONFIG", os.path.join(os.getcwd(), "config.properties")
        )

    parser = configparser.ConfigParser()
    with open(config_path, "r", encoding="utf-8") as handle:
        content = "[DEFAULT]\n" + handle.read()
    parser.read_string(content)
    values = parser["DEFAULT"]

    port = _positive_int(values.get("port", 5050), "port")
    log_level = values.get("log_level", "INFO")
    default_model = values.get(
        "default_model",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    max_workers = _positive_int(values.get("max_workers", 8), "max_workers")
    secret = values.get("secret") or None

    precision = _validate_choice(
        os.environ.get("GRAPHWISE_PRECISION", values.get("precision", "auto")),
        VALID_PRECISIONS,
        "precision",
    )
    flash_attention = _parse_bool(
        os.environ.get(
            "GRAPHWISE_FLASH_ATTENTION", values.get("flash_attention", "true")
        ),
        default=True,
    )
    max_queue_size = _positive_int(
        os.environ.get(
            "GRAPHWISE_MAX_QUEUE_SIZE", values.get("max_queue_size", 1024)
        ),
        "max_queue_size",
    )
    max_batch_tokens = _positive_int(
        os.environ.get(
            "GRAPHWISE_MAX_BATCH_TOKENS",
            values.get("max_batch_tokens", 16384),
        ),
        "max_batch_tokens",
    )
    max_batch_wait_ms = _positive_float(
        os.environ.get(
            "GRAPHWISE_MAX_BATCH_WAIT_MS",
            values.get("max_batch_wait_ms", 2.0),
        ),
        "max_batch_wait_ms",
    )
    runtime_target = _validate_choice(
        os.environ.get("GRAPHWISE_RUNTIME_TARGET", "auto"),
        VALID_RUNTIME_TARGETS,
        "runtime target",
    )

    if "GRAPHWISE_CPU_WORKERS" in os.environ:
        cpu_workers_raw = os.environ["GRAPHWISE_CPU_WORKERS"]
    elif runtime_target == "gpu":
        cpu_workers_raw = 0
    else:
        cpu_workers_raw = values.get("cpu_workers", 1)

    cpu_workers = _nonnegative_int(cpu_workers_raw, "cpu_workers")
    if runtime_target == "gpu" and cpu_workers != 0:
        raise ValueError(
            "GRAPHWISE_CPU_WORKERS must be 0 for the gpu Docker target; "
            "GPU images do not permit CPU inference workers"
        )
    if runtime_target in {"cpu", "auto"} and cpu_workers == 0:
        raise ValueError(
            "cpu_workers must be > 0 for the cpu/auto runtime target"
        )
    worker_start_timeout_seconds = _positive_float(
        os.environ.get(
            "GRAPHWISE_WORKER_START_TIMEOUT_SECONDS",
            values.get("worker_start_timeout_seconds", 300.0),
        ),
        "worker_start_timeout_seconds",
    )
    request_timeout_seconds = _positive_float(
        os.environ.get(
            "GRAPHWISE_REQUEST_TIMEOUT_SECONDS",
            values.get("request_timeout_seconds", 300.0),
        ),
        "request_timeout_seconds",
    )

    logging.getLogger().setLevel(
        getattr(logging, log_level.upper(), logging.INFO)
    )

    return AppConfig(
        port=port,
        log_level=log_level,
        default_model=default_model,
        max_workers=max_workers,
        secret=secret,
        precision=precision,
        flash_attention=flash_attention,
        max_queue_size=max_queue_size,
        max_batch_tokens=max_batch_tokens,
        max_batch_wait_ms=max_batch_wait_ms,
        cpu_workers=cpu_workers,
        runtime_target=runtime_target,
        worker_start_timeout_seconds=worker_start_timeout_seconds,
        request_timeout_seconds=request_timeout_seconds,
    )
