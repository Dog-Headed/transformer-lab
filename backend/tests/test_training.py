import time

import pytest
import torch
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config import TrainConfig
from backend.data import DATA_PATH, prepare_data, tokenize
from backend.model import TinyTransformer, parameter_count

TEXT = DATA_PATH.read_text(encoding="utf-8")[:4096]
CONFIG = {"layers": 1, "heads": 2, "hidden_size": 16, "sequence_length": 8,
          "batch_size": 4, "max_steps": 3, "eval_interval": 1, "seed": 7, "device": "cpu"}


def test_large_word_vocabulary_is_bounded_and_validation_cannot_leak():
    def word(index):
        return "word" + "".join(chr(97 + (index // (26**power)) % 26) for power in range(4))
    text = ("common " * 1000) + " ".join(word(i) for i in range(6000)) + (" heldoutonly" * 400)
    train, validation, vocabulary, data = prepare_data(text, 8, .1, "word")
    assert len(vocabulary) == 4096 and "common" in vocabulary
    assert "heldoutonly" not in vocabulary
    assert data["train_unknown_tokens"] > 0 and data["validation_unknown_tokens"] > 0
    assert int((train.tokens == 0).sum()) == data["train_unknown_tokens"]
    assert int((validation.tokens == 0).sum()) == data["validation_unknown_tokens"]
    assert prepare_data(text, 8, .1, "word")[2:] == (vocabulary, data)


def test_identity_corpus_is_deterministic_real_training_data(monkeypatch):
    from backend.data import DATASETS, read_corpus
    class Stories:
        def read_text(self, encoding):
            return "\n\n".join(["Once upon a time there was a little dog. " * 10] * 24)
    monkeypatch.setitem(DATASETS,"stories",Stories())
    first = read_corpus("stories_identity")
    assert first == read_corpus("stories_identity")
    assert "User: Who created you?\nAssistant: My creator is 狗头人." in first
    assert "我的创造者叫做狗头人。" in first
    _, _, vocabulary, _ = prepare_data(first,8,.1,"word")
    assert set(tokenize("User: Who are you? Assistant: Transformer Training Lab", "word")) <= set(vocabulary)


def test_story_dataset_selection_and_checkpoint_persist_config(client, monkeypatch):
    from backend.data import DATASETS
    class SmallStoryFile:
        def read_text(self, encoding):
            return "Once upon a time there was a little dog. The dog played in the park. " * 20
    monkeypatch.setitem(DATASETS, "stories", SmallStoryFile())
    response = client.post("/tasks", json={"config": {**CONFIG,"tokenizer":"word","dataset":"stories"}})
    assert response.status_code == 201
    job = wait(client, response.json()["id"])
    assert job["config"]["dataset"] == "stories" and job["data"]["characters"] > 1000
    loaded = client.post(f"/checkpoints/{job['checkpoint_id']}/load", json={"device":"cpu"})
    assert loaded.status_code == 201 and loaded.json()["config"]["dataset"] == "stories"
    generated = client.post(f"/tasks/{loaded.json()['id']}/generate", json={"prompt":"Once upon a time","max_new_tokens":8})
    assert generated.status_code == 200 and len(generated.json()["generated_text"]) > 0
    invalid = client.post("/tasks",json={"config":{**CONFIG,"dataset":"unknown"}})
    assert invalid.status_code == 422


def test_sentence_generation_stops_on_real_model_punctuation(client):
    task_id = create(client, tokenizer="word", max_steps=1)
    job = client.app.state.manager.jobs[task_id]
    wait(client, task_id)
    # Force the trained model's head to predict punctuation; no mock generation path.
    with torch.no_grad():
        job.model.lm_head.weight.zero_()
        job.model.lm_head.bias.fill_(-100)
        job.model.lm_head.bias[job.vocabulary.index(".")] = 100
    body = {"prompt":"First Citizen:","top_k":1,"max_new_tokens":8}
    one = client.post(f"/tasks/{task_id}/generate",json={**body,"stop_after_sentence":True}).json()
    full = client.post(f"/tasks/{task_id}/generate",json=body).json()
    assert one["generated_text"] == "." and full["generated_text"] == "."*8


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path)) as session:
        yield session


def wait(client, task_id):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        response = client.get(f"/tasks/{task_id}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] in {"completed", "stopped", "failed"}:
            assert job["status"] != "failed", job["error"]
            return job
        time.sleep(0.01)
    pytest.fail("Training did not finish within 30 seconds")


def create(client, **overrides):
    response = client.post("/tasks", json={"config": {**CONFIG, **overrides}, "text": TEXT})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_minimal_training_updates_weights_metrics_and_attention(client):
    task_id = create(client)
    job = wait(client, task_id)
    assert job["status"] == "completed" and job["step"] == 3
    assert [m["step"] for m in job["metrics"]] == [0, 1, 2, 3]
    for point in job["metrics"]:
        assert point["train_loss"] > 0 and point["validation_loss"] > 0
        assert point["perplexity"] == pytest.approx(__import__("math").exp(point["validation_loss"]))
        assert 0 <= point["accuracy"] <= 1
    manager = client.app.state.manager
    trained = manager.jobs[task_id].model
    torch.manual_seed(CONFIG["seed"])
    baseline = TinyTransformer(TrainConfig(**CONFIG), job["data"]["vocabulary_size"])
    assert any(not torch.equal(a, b) for a, b in zip(baseline.parameters(), trained.parameters()))
    weights = torch.tensor(job["attention"]["weights"])
    assert torch.allclose(weights.sum(-1), torch.ones(weights.shape[0]), atol=1e-6)
    assert weights.triu(1).count_nonzero() == 0
    assert job["checkpoint_id"]


def test_checkpoint_roundtrip_generation_and_service_restart(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        task_id = create(client)
        wait(client, task_id)
        saved = client.post(f"/tasks/{task_id}/checkpoint")
        assert saved.status_code == 201
        checkpoint_id = saved.json()["id"]
        request = {"prompt": "First", "max_new_tokens": 12, "seed": 9, "temperature": 0.8}
        before = client.post(f"/tasks/{task_id}/generate", json=request)
        assert before.status_code == 200
        assert len(before.json()["generated_text"]) == 12
        assert before.json()["text"].startswith("First")
        assert len(client.get("/checkpoints").json()) == 2
        assert client.get(f"/checkpoints/{checkpoint_id}/download").content
    with TestClient(create_app(tmp_path)) as restarted:
        assert restarted.get("/tasks").json() == []
        loaded = restarted.post(f"/checkpoints/{checkpoint_id}/load", json={"device": "cpu"})
        assert loaded.status_code == 201, loaded.text
        assert loaded.json()["status"] == "loaded" and loaded.json()["step"] == 3
        after = restarted.post(f"/tasks/{loaded.json()['id']}/generate", json=request)
        assert after.status_code == 200
        assert after.json()["generated_text"] == before.json()["generated_text"]
        assert after.json()["attention"] == before.json()["attention"]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA hardware/runtime unavailable"))])
def test_resume_restores_optimizer_and_batch_cursor_exactly(client, device):
    uninterrupted = create(client, max_steps=4, device=device)
    wait(client, uninterrupted)
    expected = {k: v.clone() for k, v in client.app.state.manager.jobs[uninterrupted].model.state_dict().items()}
    interrupted = create(client, max_steps=2, device=device)
    job = wait(client, interrupted)
    loaded = client.post(f"/checkpoints/{job['checkpoint_id']}/load", json={"device": device}).json()
    task_id = loaded["id"]
    response = client.post(f"/tasks/{task_id}/resume", json={"max_steps": 4})
    assert response.status_code == 200
    assert wait(client, task_id)["step"] == 4
    actual = client.app.state.manager.jobs[task_id].model.state_dict()
    assert all(torch.equal(expected[k], actual[k]) for k in expected)
    assert client.post(f"/tasks/{task_id}/resume", json={"max_steps": 4}).status_code == 422


def test_resource_limits_and_request_validation(client):
    for invalid in [{"heads": 3}, {"hidden_size": 1024}, {"max_steps": 99999},
                    {"sequence_length": 128, "batch_size": 32, "layers": 4, "heads": 8},
                    {"learning_rate": 0}, {"unexpected": 1}]:
        assert client.post("/tasks", json={"config": {**CONFIG, **invalid}, "text": TEXT}).status_code == 422
    assert client.post("/tasks", json={"text": "a" * 100001}).status_code == 422
    assert client.post("/tasks", json={"text": "a" * 256}).status_code == 422
    assert client.get("/tasks/missing").status_code == 404
    assert client.post("/checkpoints/not-a-valid-id/load", json={}).status_code == 404
    task_id = create(client, max_steps=2000)
    assert client.post("/tasks", json={"config": CONFIG, "text": TEXT}).status_code == 409
    assert client.post(f"/tasks/{task_id}/generate", json={"prompt": "First"}).status_code == 409
    assert client.post(f"/tasks/{task_id}/stop").status_code == 200
    job = wait(client, task_id)
    assert job["status"] == "stopped" and job["step"] < 2000 and job["checkpoint_id"]
    assert client.post(f"/tasks/{task_id}/generate", json={"prompt": "中文"}).status_code == 422
    assert client.post(f"/tasks/{task_id}/generate", json={"prompt": "First", "layer": 3}).status_code == 422
    assert client.post(f"/tasks/{task_id}/generate", json={"prompt": "First", "max_new_tokens": 129}).status_code == 422


def test_dataset_split_shift_and_causal_invariance():
    training, validation, vocabulary, data = prepare_data(TEXT, 8, 0.1)
    assert data["train_characters"] + data["validation_characters"] == len(TEXT)
    x, y = training[0]
    assert torch.equal(x[1:], y[:-1])
    split = data["train_characters"]
    assert vocabulary[int(validation[0][0][0])] == TEXT[split]
    config = TrainConfig(**CONFIG)
    model = TinyTransformer(config, len(vocabulary)).eval()
    assert sum(p.numel() for p in model.parameters()) == parameter_count(config, len(vocabulary))
    altered = x.clone()
    altered[4:] = 0
    with torch.inference_mode():
        before, _ = model(x.unsqueeze(0))
        after, _ = model(altered.unsqueeze(0))
    assert torch.equal(before[:, :4], after[:, :4])


def test_corrupt_checkpoint_and_cors(client, tmp_path):
    corrupt_id = "a" * 32
    (tmp_path / f"{corrupt_id}.pt").write_bytes(b"invalid checkpoint")
    (tmp_path / f"{corrupt_id}.json").write_text('{"id": 123, "config": {}}', encoding="utf-8")
    assert client.get("/checkpoints").json() == []
    assert client.post(f"/checkpoints/{corrupt_id}/load", json={"device": "cpu"}).status_code == 422
    response = client.options("/tasks", headers={"Origin": "http://localhost:5173",
                                                "Access-Control-Request-Method": "POST"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    denied = client.options("/tasks", headers={"Origin": "https://untrusted.example",
                                              "Access-Control-Request-Method": "POST"})
    assert denied.status_code == 400


def test_device_discovery_and_unavailable_cuda(client, monkeypatch):
    assert "cpu" in client.get("/health").json()["devices"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    response = client.post("/tasks", json={"config": {**CONFIG, "device": "cuda"}, "text": TEXT})
    assert response.status_code == 422 and "CUDA" in response.json()["detail"]
    task_id = create(client, device="auto", max_steps=1)
    assert wait(client, task_id)["device"] == "cpu"


def test_word_tokenization_prediction_and_one_word_step(client):
    assert tokenize("First Citizen: I'm ready.", "word") == ["First", "Citizen", ":", "I'm", "ready", "."]
    assert tokenize("我来到北京清华大学。", "word") == ["我", "来到", "北京", "清华大学", "。"]
    task_id = create(client, tokenizer="word", max_steps=3)
    job = wait(client, task_id)
    assert job["data"]["tokenizer"] == "word"
    body = {"prompt": "First Citizen:", "temperature": 1, "top_k": 0, "seed": 17}
    response = client.post(f"/tasks/{task_id}/predict", json=body)
    assert response.status_code == 200
    prediction = response.json()
    assert prediction["context_tokens"] == ["First", "Citizen", ":"]
    candidates = prediction["candidates"]
    assert len(candidates) == job["data"]["vocabulary_size"] - 1
    assert sum(c["probability"] for c in candidates) == pytest.approx(1, abs=1e-6)
    assert all(c["probability"] == c["model_probability"] for c in candidates)
    assert any(len(c["token"]) > 1 for c in candidates)
    # Independently confirm the displayed distribution against the trained model's logits.
    manager = client.app.state.manager
    model_job = manager.jobs[task_id]
    ids = [model_job.vocabulary.index(token) for token in prediction["context_tokens"]]
    with torch.inference_mode():
        logits, _ = model_job.model(torch.tensor([ids]))
        logits[0, -1, 0] = -float("inf")
        expected = logits[0, -1].softmax(-1)
    assert all(c["probability"] == pytest.approx(float(expected[c["token_id"]])) for c in candidates)
    greedy = {**body, "top_k": 1}
    stepped = client.post(f"/tasks/{task_id}/step", json=greedy).json()
    assert stepped["next_token"] == candidates[0]["token"]
    assert stepped["probability"] == 1
    assert stepped["prediction"]["context_tokens"] == prediction["context_tokens"] + [stepped["next_token"]]
    assert stepped["prediction"]["candidates"] != candidates
    continued = client.post(f"/tasks/{task_id}/step", json={**body, "context_tokens": stepped["prediction"]["context_tokens"]})
    assert continued.status_code == 200
    assert client.post(f"/tasks/{task_id}/predict", json={**body, "context_tokens": ["unknown-word"]}).status_code == 422
    loaded = client.post(f"/checkpoints/{job['checkpoint_id']}/load", json={"device": "cpu"}).json()
    restored = client.post(f"/tasks/{loaded['id']}/predict", json=body).json()
    assert restored["candidates"] == candidates
    assert restored["tokenizer"] == "word"
    assert client.post(f"/tasks/{loaded['id']}/generate", json={**body, "max_new_tokens": 8}).status_code == 200


def test_chinese_word_training_keeps_explicit_token_history(client):
    text = "我来到北京清华大学。我们学习人工智能和深度学习。" * 30
    response = client.post("/tasks", json={"text": text, "config": {**CONFIG, "tokenizer": "word"}})
    assert response.status_code == 201
    job = wait(client, response.json()["id"])
    body = {"prompt": "我来到北京清华大学", "top_k": 0}
    prediction = client.post(f"/tasks/{job['id']}/predict", json=body).json()
    assert prediction["context_tokens"] == ["我", "来到", "北京", "清华大学"]
    assert "清华大学" in [c["token"] for c in prediction["candidates"]]
    history = ["深度", "学习", "人工智能"]
    step = client.post(f"/tasks/{job['id']}/step", json={**body, "context_tokens": history}).json()
    assert step["prediction"]["context_tokens"] == history + [step["next_token"]]


def test_legacy_character_checkpoint_remains_loadable(client, tmp_path):
    task_id = create(client)
    job = wait(client, task_id)
    path = tmp_path / f"{job['checkpoint_id']}.pt"
    payload = torch.load(path, weights_only=True)
    payload["format_version"] = 1
    payload["config"].pop("tokenizer")
    for key in ("tokenizer", "tokenizer_version", "train_tokens", "validation_tokens", "validation_unknown_tokens"):
        payload["data"].pop(key)
    torch.save(payload, path)
    loaded = client.post(f"/checkpoints/{job['checkpoint_id']}/load", json={"device": "cpu"})
    assert loaded.status_code == 201
    assert loaded.json()["config"]["tokenizer"] == "character"
    assert client.post(f"/tasks/{loaded.json()['id']}/predict", json={"prompt": "First"}).status_code == 200


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware/runtime unavailable")
def test_cuda_train_save_load_generate_and_cpu_portability(client):
    task_id = create(client, device="cuda", max_steps=2)
    job = wait(client, task_id)
    assert job["device"] == "cuda" and job["peak_gpu_memory_mb"] > 0
    assert next(client.app.state.manager.jobs[task_id].model.parameters()).is_cuda
    body = {"prompt": "First", "max_new_tokens": 8, "seed": 19}
    before = client.post(f"/tasks/{task_id}/generate", json=body).json()
    loaded = client.post(f"/checkpoints/{job['checkpoint_id']}/load", json={"device": "cuda"}).json()
    after = client.post(f"/tasks/{loaded['id']}/generate", json=body).json()
    assert before["generated_text"] == after["generated_text"]
    cpu = client.post(f"/checkpoints/{job['checkpoint_id']}/load", json={"device": "cpu"})
    assert cpu.status_code == 201 and cpu.json()["device"] == "cpu"
    assert client.post(f"/tasks/{cpu.json()['id']}/generate", json=body).status_code == 200
