import json

import pytest

from backend.cli import main
from backend.config import GenerateRequest
from backend.data import DATA_PATH
from backend.training import TrainingManager


def test_cli_trains_persistent_model_and_generates_same_checkpoint(tmp_path, monkeypatch, capsys):
    text = DATA_PATH.read_text(encoding="utf-8")[:4096]
    monkeypatch.setattr("backend.training.read_corpus", lambda dataset: text)
    storage, output = tmp_path / "checkpoints", tmp_path / "experiment.json"
    main(["train", "--device", "cpu", "--steps", "2", "--eval-interval", "1",
          "--storage", str(storage), "--output", str(output)])
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["steps"] == 2 and result["checkpoint_generation_matches"]
    assert (storage / (result["checkpoint_id"] + ".pt")).exists()
    capsys.readouterr()
    main(["generate", "--storage", str(storage), "--checkpoint", result["checkpoint_id"],
          "--device", "cpu", "--prompt", "First Citizen:", "--max-new-tokens", "8"])
    actual = capsys.readouterr().out.strip()
    manager = TrainingManager(storage)
    try:
        job = manager.load(result["checkpoint_id"], "cpu")
        expected = manager.generate(job["id"], GenerateRequest(
            prompt="First Citizen:", max_new_tokens=8, temperature=.7, top_k=10, seed=42))
        assert actual == expected["text"].strip()
    finally:
        manager.close()


def test_cli_reports_missing_checkpoint_without_traceback(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["generate", "--storage", str(tmp_path), "--checkpoint", "0" * 32,
              "--device", "cpu", "--prompt", "First"])
    assert error.value.code == 1
    message = capsys.readouterr().err
    assert message.startswith("Error:") and "Traceback" not in message
