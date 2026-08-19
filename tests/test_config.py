from pathlib import Path

import pytest

from graphwise_transformer.config import load_config


BASE = """\
port=5050
log_level=INFO
default_model=test-model
max_workers=4
precision=auto
flash_attention=true
max_queue_size=256
max_batch_tokens=4096
max_batch_wait_ms=3
cpu_workers=2
worker_start_timeout_seconds=30
request_timeout_seconds=40
"""


def _write_config(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "config.properties"
    path.write_text(BASE + extra, encoding="utf-8")
    return path


def test_defaults_and_scheduler_fields(tmp_path, monkeypatch):
    for name in (
        "GRAPHWISE_PRECISION",
        "GRAPHWISE_FLASH_ATTENTION",
        "GRAPHWISE_MAX_QUEUE_SIZE",
        "GRAPHWISE_MAX_BATCH_TOKENS",
        "GRAPHWISE_MAX_BATCH_WAIT_MS",
        "GRAPHWISE_CPU_WORKERS",
        "GRAPHWISE_RUNTIME_TARGET",
    ):
        monkeypatch.delenv(name, raising=False)
    config = load_config(str(_write_config(tmp_path)))
    assert config.precision == "auto"
    assert config.flash_attention is True
    assert config.max_queue_size == 256
    assert config.max_batch_tokens == 4096
    assert config.max_batch_wait_ms == 3
    assert config.cpu_workers == 2


@pytest.mark.parametrize("precision", ["auto", "fp16", "bf16", "fp32"])
def test_supported_precision_override(tmp_path, monkeypatch, precision):
    monkeypatch.setenv("GRAPHWISE_PRECISION", precision)
    config = load_config(str(_write_config(tmp_path)))
    assert config.precision == precision


@pytest.mark.parametrize("precision", ["int8", "fp8", "float16", "cuda"])
def test_unsupported_precision_is_rejected(tmp_path, monkeypatch, precision):
    monkeypatch.setenv("GRAPHWISE_PRECISION", precision)
    with pytest.raises(ValueError, match="Invalid precision"):
        load_config(str(_write_config(tmp_path)))


def test_flash_attention_can_be_disabled_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHWISE_FLASH_ATTENTION", "false")
    config = load_config(str(_write_config(tmp_path)))
    assert config.flash_attention is False


def test_invalid_boolean_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHWISE_FLASH_ATTENTION", "maybe")
    with pytest.raises(ValueError, match="Invalid boolean"):
        load_config(str(_write_config(tmp_path)))


def test_gpu_target_defaults_cpu_workers_to_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHWISE_RUNTIME_TARGET", "gpu")
    monkeypatch.delenv("GRAPHWISE_CPU_WORKERS", raising=False)
    config = load_config(str(_write_config(tmp_path)))
    assert config.runtime_target == "gpu"
    assert config.cpu_workers == 0


def test_gpu_target_rejects_cpu_workers(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHWISE_RUNTIME_TARGET", "gpu")
    monkeypatch.setenv("GRAPHWISE_CPU_WORKERS", "1")
    with pytest.raises(ValueError, match="must be 0 for the gpu Docker target"):
        load_config(str(_write_config(tmp_path)))


def test_cpu_target_defaults_to_configured_cpu_workers(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHWISE_RUNTIME_TARGET", "cpu")
    monkeypatch.delenv("GRAPHWISE_CPU_WORKERS", raising=False)
    config = load_config(str(_write_config(tmp_path)))
    assert config.runtime_target == "cpu"
    assert config.cpu_workers == 2


def test_cpu_target_rejects_zero_cpu_workers(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHWISE_RUNTIME_TARGET", "cpu")
    monkeypatch.setenv("GRAPHWISE_CPU_WORKERS", "0")
    with pytest.raises(ValueError, match="cpu_workers must be > 0"):
        load_config(str(_write_config(tmp_path)))
