from graphwise_transformer.model import _batch_cost, _target_token_span


def test_unpadded_batch_cost_is_sum_of_real_tokens():
    assert _batch_cost([4, 8, 2], unpadded=True) == 14


def test_sdpa_batch_cost_accounts_for_padding():
    assert _batch_cost([4, 8, 2], unpadded=False) == 24


def test_target_token_span_uses_overlapping_content_tokens():
    offsets = [(0, 0), (0, 4), (5, 9), (0, 0)]
    assert _target_token_span(offsets, (2, 8), item_index=0) == (1, 3)


def test_target_token_span_preserves_legacy_truncation_semantics():
    offsets = [(0, 0), (0, 4), (5, 9), (0, 0)]
    assert _target_token_span(offsets, (2, 15), item_index=0) == (1, 3)


def test_target_token_span_accepts_leading_unrepresented_whitespace():
    # Fast tokenizers do not assign offsets to whitespace.  A Graphwise span may
    # still begin in that whitespace and should map to the first overlapping token.
    offsets = [(0, 0), (1, 6), (7, 11), (0, 0)]
    assert _target_token_span(offsets, (0, 2), item_index=0) == (1, 2)


def test_visible_worker_devices_uses_all_visible_gpus(monkeypatch):
    import graphwise_transformer.model as model_module

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 3

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setattr(model_module, "torch", FakeTorch())
    assert model_module._visible_worker_devices(cpu_workers=4) == [
        "cuda:0",
        "cuda:1",
        "cuda:2",
    ]


def test_visible_worker_devices_falls_back_to_cpu_workers(monkeypatch):
    import graphwise_transformer.model as model_module

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setattr(model_module, "torch", FakeTorch())
    assert model_module._visible_worker_devices(cpu_workers=3) == ["cpu", "cpu", "cpu"]


def test_gpu_runtime_never_falls_back_to_cpu(monkeypatch):
    import pytest
    import graphwise_transformer.model as model_module

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setattr(model_module, "torch", FakeTorch())
    with pytest.raises(RuntimeError, match="requires at least one visible CUDA device"):
        model_module._visible_worker_devices(cpu_workers=0, runtime_target="gpu")


def test_gpu_runtime_rejects_cpu_workers(monkeypatch):
    import pytest
    import graphwise_transformer.model as model_module

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setattr(model_module, "torch", FakeTorch())
    with pytest.raises(RuntimeError, match="does not permit CPU workers"):
        model_module._visible_worker_devices(cpu_workers=1, runtime_target="gpu")


def test_cpu_runtime_ignores_visible_gpus(monkeypatch):
    import graphwise_transformer.model as model_module

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 4

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setattr(model_module, "torch", FakeTorch())
    assert model_module._visible_worker_devices(cpu_workers=1, runtime_target="cpu") == ["cpu"]


def test_token_encode_uses_supported_api_and_temporarily_disables_unpadding():
    import torch

    from graphwise_transformer.model import _encode_token_outputs

    class FakeTransformer:
        unpad_inputs = None

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            assert kwargs["padding"] is False
            assert kwargs["truncation"] is True
            assert kwargs["max_length"] == 128
            assert kwargs["add_special_tokens"] is True
            return {"input_ids": [[101, 11, 102], [101, 12, 13, 102]]}

    class FakeModel:
        max_seq_length = 128
        tokenizer = FakeTokenizer()

        def __init__(self):
            self.transformer = FakeTransformer()
            self.encode_kwargs = None

        def __getitem__(self, index):
            assert index == 0
            return self.transformer

        def encode(self, texts, **kwargs):
            assert self.transformer.unpad_inputs is False
            self.encode_kwargs = kwargs
            return [
                torch.zeros((3, 2), dtype=torch.float32),
                torch.zeros((4, 2), dtype=torch.float32),
            ]

    model = FakeModel()
    outputs = _encode_token_outputs(model, ["a", "b"], batch_size=2)

    assert model.transformer.unpad_inputs is None
    assert model.encode_kwargs["output_value"] == "token_embeddings"
    assert model.encode_kwargs["convert_to_numpy"] is False
    assert model.encode_kwargs["convert_to_tensor"] is False
    assert len(outputs) == 2
    assert outputs[0].input_ids == [101, 11, 102]
    assert outputs[1].input_ids == [101, 12, 13, 102]


def test_token_encode_restores_unpadding_when_encode_fails():
    import pytest

    from graphwise_transformer.model import _encode_token_outputs

    class FakeTransformer:
        unpad_inputs = None

    class FakeModel:
        max_seq_length = 128

        def __init__(self):
            self.transformer = FakeTransformer()

        def __getitem__(self, index):
            return self.transformer

        def encode(self, texts, **kwargs):
            assert self.transformer.unpad_inputs is False
            raise RuntimeError("boom")

    model = FakeModel()
    with pytest.raises(RuntimeError, match="boom"):
        _encode_token_outputs(model, ["a", "b"], batch_size=2)
    assert model.transformer.unpad_inputs is None


def test_output_kind_batches_token_and_offset_together_but_not_sentence():
    from graphwise_transformer.model import EmbeddingType, _WorkItem, _output_kind

    sentence = _WorkItem("s", EmbeddingType.SENTENCE.value, ("a",), None)
    token = _WorkItem("t", EmbeddingType.TOKEN.value, ("a",), None)
    offset = _WorkItem("o", EmbeddingType.OFFSET.value, ("a",), ((0, 1),))

    assert _output_kind(sentence) == "sentence"
    assert _output_kind(token) == "token"
    assert _output_kind(offset) == "token"
