"""Local CLI using the same real training and inference implementation as the API."""
import argparse
import json
import os
from pathlib import Path
import sys

DEFAULT_STORAGE = Path(__file__).resolve().parent / "runs"


def build_parser():
    from .benchmark import configure_parser

    parser = argparse.ArgumentParser(prog="transformer-lab")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Show Python, CUDA, and checkpoint environment")
    serve = commands.add_parser("serve", help="Start the FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--storage", type=Path, default=DEFAULT_STORAGE)
    train = commands.add_parser("train", help="Train and record a reproducible experiment")
    configure_parser(train)
    train.set_defaults(storage=DEFAULT_STORAGE, output=Path("experiments/local-run.json"))
    generate = commands.add_parser("generate", help="Generate text from a saved checkpoint")
    generate.add_argument("--checkpoint", required=True, help="Checkpoint ID, without .pt")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--storage", type=Path, default=DEFAULT_STORAGE)
    generate.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    generate.add_argument("--max-new-tokens", type=int, default=64)
    generate.add_argument("--temperature", type=float, default=.7)
    generate.add_argument("--top-k", type=int, default=10)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--stop-after-sentence", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            from .training import TrainingManager
            info = {"python": sys.executable, "project": str(DEFAULT_STORAGE.parent.parent),
                    "storage": str(DEFAULT_STORAGE),
                    "checkpoints": len(list(DEFAULT_STORAGE.glob("*.pt"))),
                    **TrainingManager.hardware()}
            print(json.dumps(info, ensure_ascii=False, indent=2))
        elif args.command == "serve":
            import uvicorn
            os.environ["TRAINING_STORAGE_DIR"] = str(args.storage.resolve())
            uvicorn.run("backend.app:app", host=args.host, port=args.port, workers=1)
        elif args.command == "train":
            from .benchmark import run
            run(args)
        else:
            from .config import GenerateRequest
            from .training import TrainingManager
            manager = TrainingManager(args.storage)
            try:
                loaded = manager.load(args.checkpoint, args.device)
                generated = manager.generate(loaded["id"], GenerateRequest(
                    prompt=args.prompt, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_k=args.top_k, seed=args.seed,
                    stop_after_sentence=args.stop_after_sentence))
                print(generated["text"])
            finally:
                manager.close()
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    except Exception as exc:
        from .training import ServiceError
        if isinstance(exc, ServiceError):
            parser.exit(1, f"Error: {exc.detail}\n")
        raise


if __name__ == "__main__":
    main()
