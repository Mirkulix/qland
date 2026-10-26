"""Sizes and accuracies the platform reports are measured: packed files, held-out test splits.

Runs offline, except MNIST, which uses an existing download ($IGQK_DATA_DIR) and is skipped otherwise."""

import asyncio
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "igqk"))

from services.packing import is_packed, load_packed, save_packed  # noqa: E402


def test_packed_ternary_weights_take_two_bits(tmp_path):
    from igqk import IGQKOptimizer, TernaryProjector
    model = nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 10))
    IGQKOptimizer(model.parameters(), projector=TernaryProjector()).compress(model)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    saved = save_packed(model.state_dict(), str(tmp_path / "m.pt"))
    assert saved["file_bytes"] == os.path.getsize(tmp_path / "m.pt")
    assert saved["packed_tensors"] >= 2
    ratio = saved["fp32_bytes"] / saved["file_bytes"]
    assert 15 < ratio <= 16, ratio
    ckpt = torch.load(tmp_path / "m.pt", weights_only=False)
    assert is_packed(ckpt)
    for k, v in load_packed(ckpt).items():
        assert torch.equal(v, before[k]), k


def test_dense_weights_are_not_shrunk(tmp_path):
    model = nn.Linear(256, 256)
    saved = save_packed(model.state_dict(), str(tmp_path / "d.pt"))
    assert saved["packed_tensors"] == 0
    assert saved["file_bytes"] >= saved["fp32_bytes"]


def test_a_failed_hub_search_is_an_error(monkeypatch):
    from fastapi.testclient import TestClient
    from services import huggingface_service

    def down(**kw):
        raise ConnectionError("hub unreachable")
    monkeypatch.setattr(huggingface_service, "list_models", down)
    import main
    r = TestClient(main.app).get("/api/models/search/huggingface", params={"query": "bert"})
    assert r.status_code == 502, r.text


def _train(tmp_path, monkeypatch, **cfg):
    from api import training
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    config = training.TrainingConfig(**{"job_name": "t", "epochs": 1, "batch_size": 16, **cfg})
    training.training_jobs["j"] = {"config": config.dict(), "status": "queued", "progress": 0.0, "current_epoch": 0}
    asyncio.run(training.run_training_job("j", config))
    return training.training_jobs.pop("j")


@pytest.mark.parametrize("arch", ["gpt2", "bert"])
def test_language_models_train_on_text_and_report_test_accuracy(tmp_path, monkeypatch, arch):
    text = tmp_path / "corpus.txt"
    text.write_text("the quick brown fox jumps over the lazy dog. " * 200)
    job = _train(tmp_path, monkeypatch, dataset_id=str(text), architecture=arch, optimizer="adam", learning_rate=0.01)
    assert job["status"] == "completed", job.get("message")
    assert 0 <= job["test_accuracy"] <= 100
    assert 0 <= job["compressed_test_accuracy"] <= 100
    assert os.path.exists(job["checkpoint_path"])
    assert job["compressed_size_mb"] == round(os.path.getsize(job["compressed_path"]) / (1024 * 1024), 4)
    assert job["compressed_size_mb"] < job["model_size_mb"]


def test_gpt_learns_next_byte(tmp_path, monkeypatch):
    # A repeated pattern is predictable: a model that learns the next byte beats chance by far.
    text = tmp_path / "corpus.txt"
    text.write_text("abcdefgh" * 2000)
    job = _train(tmp_path, monkeypatch, dataset_id=str(text), architecture="gpt2", optimizer="adam", learning_rate=0.01, auto_compress=False, epochs=3)
    assert job["status"] == "completed", job.get("message")
    assert job["test_accuracy"] > 50  # chance is 1/256


def test_an_unavailable_dataset_fails_the_job(tmp_path, monkeypatch):
    job = _train(tmp_path, monkeypatch, dataset_id="imagenet", architecture="custom")
    assert job["status"] == "failed"
    assert "not available" in job["message"]


@pytest.mark.skipif(not os.path.isdir(os.path.join(os.environ.get("IGQK_DATA_DIR", "/nonexistent"), "MNIST")), reason="MNIST not downloaded (set IGQK_DATA_DIR)")
def test_mnist_trains_on_real_images(tmp_path, monkeypatch):
    job = _train(tmp_path, monkeypatch, dataset_id="mnist", architecture="custom", optimizer="adam", learning_rate=0.001, max_samples=6000)
    assert job["status"] == "completed", job.get("message")
    assert job["test_accuracy"] > 80  # chance is 10%
    print("MNIST test accuracy", job["test_accuracy"], "-> compressed", job["compressed_test_accuracy"],
          "| size", job["model_size_mb"], "MB ->", job["compressed_size_mb"], "MB")
