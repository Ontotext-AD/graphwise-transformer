from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_target_requirements_are_split_and_pinned():
    cpu = (ROOT / "requirements_cpu.in").read_text(encoding="utf-8")
    gpu = (ROOT / "requirements_gpu.in").read_text(encoding="utf-8")

    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in cpu
    assert "torch==2.9.1+cpu" in cpu
    assert "flash-attn" not in cpu

    assert "--extra-index-url https://download.pytorch.org/whl/cu128" in gpu
    assert "torch==2.9.1+cu128" in gpu
    assert "transformers==5.14.1" in gpu
    assert "sentence-transformers==5.6.1" in gpu
    assert "protobuf==7.35.1" in gpu
    assert "flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" in gpu
    assert "sha256=4e2f9e39313266b1544b68138b15b91ee6221eccf14f7902b7c6620351340810" in gpu


def test_docker_has_only_cpu_and_gpu_runtime_targets():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM runtime-base AS cpu" in dockerfile
    assert "FROM runtime-base AS gpu" in dockerfile
    assert "requirements_cpu.txt" in dockerfile
    assert "requirements_gpu.txt" in dockerfile


def test_cpu_target_uses_cpu_requirements_and_one_worker_default():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    cpu = dockerfile.split("FROM runtime-base AS cpu", 1)[1].split(
        "FROM runtime-base AS gpu", 1
    )[0]
    assert "python -m pip install -r requirements_cpu.txt" in cpu
    assert "GRAPHWISE_RUNTIME_TARGET=cpu" in cpu
    assert "requirements_gpu.txt" not in cpu
    assert "cuda-toolkit" not in cpu


def test_gpu_target_uses_gpu_requirements_and_disables_cpu_workers():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    gpu = dockerfile.split("FROM runtime-base AS gpu", 1)[1]
    assert "python -m pip install -r requirements_gpu.txt" in gpu
    assert "GRAPHWISE_RUNTIME_TARGET=gpu" in gpu
    assert "cuda-toolkit" not in gpu
    assert "nvcc" not in gpu
    assert "pip wheel" not in gpu


def test_model_does_not_use_output_value_none_for_batched_inference():
    source = (ROOT / "graphwise_transformer" / "model.py").read_text(encoding="utf-8")
    assert 'output_value=None,' not in source
    assert 'output_value="sentence_embedding"' in source
    assert 'cu_seq_lens_q' not in source
    assert 'output_value="token_embeddings"' in source
    assert 'transformer_module.unpad_inputs = False' in source
    assert 'transformer_module.unpad_inputs = None' in source
