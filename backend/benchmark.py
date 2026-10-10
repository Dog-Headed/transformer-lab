"""Reproducible CPU/CUDA experiment; writes observed results, never preset metrics."""
import argparse
from contextlib import nullcontext
import json
import platform
import tempfile
import time
from pathlib import Path

import torch

from .config import GenerateRequest, TrainConfig, TrainRequest
from .training import TrainingManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--size", choices=["tiny", "medium"], default="tiny")
    parser.add_argument("--tokenizer", choices=["character", "word"], default="character")
    parser.add_argument("--dataset", choices=["shakespeare", "stories", "stories_identity"], default="shakespeare")
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--storage", type=Path, help="Keep real checkpoints in this directory, e.g. backend/runs")
    parser.add_argument("--output", type=Path, default=Path("experiments/cpu-baseline.json"))
    args = parser.parse_args()
    config = TrainConfig(max_steps=args.steps, device=args.device, tokenizer=args.tokenizer, dataset=args.dataset,
                         learning_rate=args.learning_rate, eval_interval=args.eval_interval,
                         **({"layers": 4, "hidden_size": 128, "sequence_length": 64} if args.size == "medium" else {}))
    with (nullcontext(args.storage) if args.storage else tempfile.TemporaryDirectory()) as directory:
        manager = TrainingManager(Path(directory))
        try:
            started = time.perf_counter()
            job = manager.create(TrainRequest(config=config))
            while job["status"] in {"queued", "running", "stopping"}:
                time.sleep(0.05)
                job = manager.snapshot(job["id"])
            if job["status"] != "completed":
                raise RuntimeError(job["error"] or job["stop_reason"])
            wall_seconds = time.perf_counter() - started
            prompt = "Once upon a time" if args.dataset.startswith("stories") else "First Citizen:"
            temperature, top_k = (0.7, 10) if args.dataset.startswith("stories") else (0.8, 20)
            options = GenerateRequest(prompt=prompt, max_new_tokens=100, temperature=temperature, top_k=top_k)
            generated = manager.generate(job["id"], options)
            loaded = manager.load(job["checkpoint_id"], job["device"])
            restored = manager.generate(loaded["id"], options)
            if generated["text"] != restored["text"]:
                raise RuntimeError("Checkpoint generation roundtrip mismatch")
            samples = []
            if args.dataset.startswith("stories"):
                for sample_prompt in ("Once upon a time", "One day", "The little girl", "The dog"):
                    sample = manager.generate(job["id"], options.model_copy(update={"prompt": sample_prompt}))
                    samples.append({"prompt": sample_prompt, "text": sample["text"], "seed":42})
            identity_samples = []
            if args.dataset == "stories_identity":
                for question in ("Who are you?", "Who created you?", "你是谁？", "你的创造者是谁？", "你好"):
                    sample = manager.generate(job["id"], GenerateRequest(prompt=f"User: {question}\nAssistant:",
                        max_new_tokens=64, temperature=.5, top_k=1, stop_after_sentence=True))
                    identity_samples.append({"question":question,"answer":sample["generated_text"].strip(),"top_k":1,"temperature":.5})
            result = {"experiment": f"{args.dataset} {config.tokenizer}-level {job['device']} {args.size} experiment", "config": config.model_dump(),
                "environment": {"python": platform.python_version(), "pytorch": str(torch.__version__),
                    "platform": platform.platform(), "processor": platform.processor(), "device": job["device"], "threads": 2,
                    "gpu_name": torch.cuda.get_device_name(0) if job["device"] == "cuda" else None},
                "data": job["data"], "parameters": job["parameter_count"], "steps": job["step"],
                "training_seconds": job["elapsed_seconds"], "wall_seconds_including_checkpoint": wall_seconds,
                "peak_gpu_memory_mb": job["peak_gpu_memory_mb"],
                "initial": job["metrics"][0], "final": job["metrics"][-1], "metrics": job["metrics"],
                "generation": {"prompt": generated["prompt"], "generated_text": generated["generated_text"],
                    "seed": 42, "temperature": temperature, "top_k": top_k},
                "samples": samples, "identity_samples":identity_samples,
                "checkpoint_id": job["checkpoint_id"] if args.storage else None,
                "checkpoint_generation_matches": True,
                "limitations": "Single seed; fixed train evaluation subset (8 batches); full held-out validation windows; small corpus; no test set or general language quality claim."}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({k: result[k] for k in ("parameters", "steps", "training_seconds", "initial", "final")}, indent=2))
        finally:
            manager.close()


if __name__ == "__main__":
    main()
