import grpc

from graphwise_transformer.model import Embedding, EmbeddingType
from graphwise_transformer.proto import transformer_pb2 as pb
from graphwise_transformer.server import InferenceService


class _FakeModel:
    def embed(self, embedding_type, texts, char_offsets=None):
        if embedding_type == EmbeddingType.TOKEN:
            return [
                [Embedding(string="tok", embedding=[float(i), 1.0])]
                for i, _text in enumerate(texts)
            ]
        if embedding_type == EmbeddingType.OFFSET:
            assert char_offsets is not None
        return [
            [Embedding(string=text, embedding=[float(i), 2.0])]
            for i, text in enumerate(texts)
        ]


class _FakeRegistry:
    def get(self, model_name):
        if model_name != "model":
            raise KeyError(model_name)
        return _FakeModel()


class _Abort(Exception):
    def __init__(self, code, details):
        self.code = code
        self.details = details


class _Context:
    def abort(self, code, details):
        raise _Abort(code, details)


def test_sentence_rpc_keeps_proto_contract():
    service = InferenceService(_FakeRegistry())
    response = service.EmbedSentence(
        pb.SentenceRequest(model_name="model", texts=["a", "b"]), _Context()
    )
    assert len(response.embeddings) == 2
    assert list(response.embeddings[1].embedding) == [1.0, 2.0]


def test_token_rpc_keeps_proto_contract():
    service = InferenceService(_FakeRegistry())
    response = service.EmbedTokens(
        pb.TokenRequest(model_name="model", texts=["a"]), _Context()
    )
    assert len(response.results) == 1
    assert response.results[0].tokens[0].string == "tok"


def test_offset_rpc_keeps_proto_contract():
    service = InferenceService(_FakeRegistry())
    response = service.EmbedWithOffsets(
        pb.OffsetRequest(
            model_name="model",
            inputs=[pb.TextWithOffsets(text="abcd", start=0, end=2)],
        ),
        _Context(),
    )
    assert len(response.embeddings) == 1
    assert response.embeddings[0].string == "abcd"


def test_missing_model_maps_to_not_found():
    service = InferenceService(_FakeRegistry())
    try:
        service.EmbedSentence(
            pb.SentenceRequest(model_name="missing", texts=["x"]), _Context()
        )
    except _Abort as exc:
        assert exc.code == grpc.StatusCode.NOT_FOUND
    else:
        raise AssertionError("Expected context.abort")
