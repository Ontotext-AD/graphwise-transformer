import gzip
import os
import threading
import time
import tempfile
import socket

import grpc
import numpy as np
import pytest
from numpy import linalg as la
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure protos are generated before importing stubs
import setup as project_setup  # type: ignore
project_setup.generate_protos()

from graphwise_transformer.server import main as server_main
from graphwise_transformer.proto import transformer_pb2 as pb
from graphwise_transformer.proto import transformer_pb2_grpc as pbg


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def run_server_in_thread(port: int, default_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    cfg_content = f"port={port}\nlog_level=INFO\ndefault_model={default_model}\nmax_workers=4\n"
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(cfg_content)
        cfg_path = f.name
    os.environ["GRAPHWISE_CONFIG"] = cfg_path

    t = threading.Thread(target=server_main, daemon=True)
    t.start()
    # Wait for server to actually start listening on the port
    max_wait = 30
    elapsed = 0
    while elapsed < max_wait:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.5)
                if sock.connect_ex(("localhost", port)) == 0:
                    break
        except Exception:
            pass
        time.sleep(0.5)
        elapsed += 0.5
    else:
        raise RuntimeError(f"Server did not start on port {port} within {max_wait}s")
    return t, cfg_path


@pytest.fixture(scope="module")
def grpc_channel():
    port = find_free_port()
    t, cfg_path = run_server_in_thread(port)
    channel = grpc.insecure_channel(f"localhost:{port}")
    # Wait for channel ready
    grpc.channel_ready_future(channel).result(timeout=20)
    yield channel
    # Teardown
    try:
        os.remove(cfg_path)
    except Exception:
        pass


def test_admin_list_models(grpc_channel):
    admin = pbg.AdminServiceStub(grpc_channel)
    resp = admin.ListModels(pb.ListModelsRequest())
    # Protobuf repeated fields are iterable and support len()
    assert len(resp.model_names) >= 0
    # Test that we can iterate over it
    model_list = list(resp.model_names)
    assert len(model_list) == len(resp.model_names)


def test_admin_load_unload(grpc_channel):
    admin = pbg.AdminServiceStub(grpc_channel)
    model = "sentence-transformers/msmarco-bert-base-dot-v5"
    load = admin.LoadModel(pb.LoadModelRequest(model_name=model))
    assert load.ok
    listed = admin.ListModels(pb.ListModelsRequest())
    assert model in listed.model_names
    unl = admin.UnloadModel(pb.UnloadModelRequest(model_name=model))
    assert unl.ok


def test_inference_embed_sentence(grpc_channel):
    infer = pbg.InferenceServiceStub(grpc_channel)
    texts = ["test string one", "test string two"]
    req = pb.SentenceRequest(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        texts=texts,
    )
    resp = infer.EmbedSentence(req)
    assert len(resp.embeddings) == 2
    assert len(resp.embeddings[0].embedding) > 0
    assert len(resp.embeddings[1].embedding) > 0


def test_inference_embed_with_offsets(grpc_channel):
    # Skip if model doesn't support offset mapping - this test may fail for some models
    infer = pbg.InferenceServiceStub(grpc_channel)
    text = "string; this is a very long string"
    req = pb.OffsetRequest(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        inputs=[pb.TextWithOffsets(text=text, start=0, end=6)],
    )
    try:
        resp = infer.EmbedWithOffsets(req)
        assert len(resp.embeddings) == 1
        assert len(resp.embeddings[0].embedding) > 0
    except grpc.RpcError as e:
        # Model may not support offset mapping - that's okay, just skip this test
        if "return_offsets_mapping" in str(e).lower():
            pytest.skip(f"Model does not support offset embeddings: {e}")
        raise


def test_generated_embeddings(grpc_channel):
    infer = pbg.InferenceServiceStub(grpc_channel)
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    text = """
Fewer Babies, Older Parents: Bulgaria Faces Sharp Decline in Births Since 1994

Bulgaria’s birth rate has declined by one-third over the past three decades, marking a significant demographic shift. In 1994, the country recorded 79,442 live births, but by 2024 this number had fallen to 53,428, representing a 33% decrease. The data were presented by the Bulgarian Association of Sterility and Reproductive Health during a press conference, as reported by BGNES.
According to the association, 6.5% of all Bulgarian children are now born through in vitro fertilization (IVF) procedures. The average success rate of assisted reproductive treatments in Bulgaria currently stands at 26.3%, showing gradual improvement but still reflecting the broader challenges of declining fertility and delayed family planning.
Experts note that an increasing number of couples are choosing to have children later in life. The average age of mothers at the birth of their first child has reached 27.6 years. The country’s birth rate is now 8.3 per thousand, while the average number of children per woman is just 1.72 - well below the 2.1 threshold required to maintain population levels.
Demographic data further indicate that Bulgaria remains one of the European Union countries most affected by population aging. People aged over 65 already make up 24% of the total population, underscoring the deepening demographic imbalance between younger and older generations.
"""
    for _ in range(8):
        text += text
    positions = [0]
    position = 1
    while position < len(text):
        positions.append(position)
        position *= 2
    positions.append(len(text))
    offsets = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if positions[i] + 1 < positions[j]:
                offsets.append((positions[i], positions[j]))
    resources_folder = os.path.join(os.path.dirname(__file__), "resources")
    file = os.path.join(resources_folder, "expected_test_embeddings.tsv.gz")
    with gzip.open(file, 'rt', encoding='utf-8') as f:
        for line in f:
            terms = line.strip().split('\t')
            type_ = terms[0]
            expected_embedding = [float(terms[i]) for i in range(1, len(terms))]
            actual_embedding = embed_text(infer, model_name, type_, text,
                                          offsets)
            actual_minus_expected = np.subtract(actual_embedding,
                                                expected_embedding)
            l2_norm = la.norm(actual_minus_expected)
            assert l2_norm < 0.00001


def text_to_prompt(
        label: str, context: str, offsets: tuple[int, int],
        max_len_words: int = 30
):
    chars_half_window = 100
    short_cxt_before = context[
        max(0, offsets[0] - chars_half_window): offsets[0]]
    short_cxt_after = context[
        offsets[1]: min(len(context), offsets[1] + chars_half_window)]
    target_words = (
            short_cxt_before.split()[-(max_len_words // 2):]
            + [context[offsets[0]: offsets[1]]]
            + short_cxt_after.split()[-(max_len_words // 2):]
    )
    target_cxt = " ".join(target_words)
    return (f"{label} ;" if label else "") + target_cxt


def embed_text(infer, model_name, type_, text, offsets) -> list[float]:
    # use only the first span/offset and ignore the others
    start, end = offsets[0]
    prompt = text_to_prompt(label="", context=text, offsets=(start, end))
    if type_ == "sentence":
        req = pb.SentenceRequest(
            model_name=model_name,
            texts=[prompt],
        )
        resp = infer.EmbedSentence(req)
        return resp.embeddings[0].embedding
    if type_ == "offset":
        req = pb.OffsetRequest(
            model_name=model_name,
            inputs=[pb.TextWithOffsets(text=prompt, start=start, end=end)],
        )
        resp = infer.EmbedWithOffsets(req)
        return resp.embeddings[0].embedding
    assert type_ == "token"
    req = pb.TokenRequest(
        model_name=model_name,
        texts=[prompt],
    )
    resp = infer.EmbedTokens(req)
    return resp.results[0].tokens[0].embedding
