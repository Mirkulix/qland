"""Deployment serves a model for real: `predict` runs it, and only a servable model is deployed.

Runs offline: a tiny BERT is built locally and saved the way the compression service saves
(`model_state_dict` + `metadata.original_identifier`, here a local directory)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient  # noqa: E402
from transformers import BertConfig, BertModel, BertTokenizerFast  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    arch = tmp_path / "tiny-bert"
    arch.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world"]
    (arch / "vocab.txt").write_text("\n".join(vocab) + "\n")
    BertTokenizerFast(vocab_file=str(arch / "vocab.txt")).save_pretrained(arch)
    config = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)
    model = BertModel(config)
    model.save_pretrained(arch)
    (tmp_path / "compressed_models").mkdir()
    torch.save({"model_state_dict": model.state_dict(), "metadata": {"original_identifier": str(arch)}}, tmp_path / "compressed_models" / "tiny_compressed.pt")
    torch.save(model.state_dict(), tmp_path / "compressed_models" / "bare.pt")
    # As the compression service saves now: ternary weights 2-bit packed, the rest as they are.
    from services.packing import save_packed
    ternary = {k: (v.sign() * 0.5 if k.endswith("dense.weight") else v) for k, v in model.state_dict().items()}
    save_packed(ternary, str(tmp_path / "compressed_models" / "tiny_packed.pt"), metadata={"original_identifier": str(arch)})
    monkeypatch.chdir(tmp_path)
    from api import models
    models._loaded.clear()
    import main
    return TestClient(main.app)


def test_predict_runs_the_model(client):
    r = client.post("/api/models/tiny_compressed.pt/predict", json={"text": "hello world"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["embedding_dim"] == 16 and len(body["embedding"]) == 16 and body["tokens"] == 4
    again = client.post("/api/models/tiny_compressed.pt/predict", json={"text": "hello world"}).json()
    assert again["embedding"] == body["embedding"]


def test_a_packed_model_is_served(client):
    r = client.post("/api/models/tiny_packed.pt/predict", json={"text": "hello world"})
    assert r.status_code == 200, r.text
    assert r.json()["embedding_dim"] == 16


def test_only_a_servable_model_is_deployed(client):
    r = client.post("/api/deployment/deploy", json={"model_id": "tiny_compressed.pt", "name": "t", "target": "igqk_api"})
    assert r.status_code == 200, r.text
    endpoint = r.json()["endpoint_url"]
    assert client.post(endpoint, json={"text": "hello"}).status_code == 200
    bare = client.post("/api/deployment/deploy", json={"model_id": "bare.pt", "name": "b", "target": "igqk_api"})
    assert bare.status_code == 422 and "no architecture" in bare.text
    missing = client.post("/api/deployment/deploy", json={"model_id": "nope.pt", "name": "n", "target": "igqk_api"})
    assert missing.status_code == 404
    cloud = client.post("/api/deployment/deploy", json={"model_id": "tiny_compressed.pt", "name": "c", "target": "aws_sagemaker"})
    assert cloud.status_code == 501


def test_dataset_search_asks_the_hub(client, monkeypatch):
    from types import SimpleNamespace
    import huggingface_hub
    asked = {}

    def fake_list_datasets(search, limit):
        asked.update(search=search, limit=limit)
        return [SimpleNamespace(id="org/imdb-like", description="reviews", tags=["task_categories:text-classification"])]

    # The Hub is replaced only here (no network in tests); the endpoint calls list_datasets.
    monkeypatch.setattr(huggingface_hub, "list_datasets", fake_list_datasets)
    r = client.get("/api/datasets/search/huggingface", params={"query": "imdb", "limit": 3})
    assert r.status_code == 200, r.text
    assert asked == {"search": "imdb", "limit": 3}
    assert r.json()[0]["id"] == "org/imdb-like" and r.json()[0]["task"] == "text-classification" and r.json()[0]["size"] is None
