from __future__ import annotations

import argparse
import json

from pre.train.trainer import Trainer


def train_main() -> None:
    parser = argparse.ArgumentParser(prog="pre-train")
    parser.add_argument("--dataset", default="nyc_taxi")
    parser.add_argument("--model", default="dummy")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--context-length", type=int, default=168)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--artifact-root", default="artifacts")
    args = parser.parse_args()

    try:
        result = Trainer().train(
            dataset=args.dataset,
            model=args.model,
            horizon=args.horizon,
            context_length=args.context_length,
            stride=args.stride,
            artifact_root=args.artifact_root,
        )
    except ValueError as err:
        print(json.dumps({"status": "error", "message": str(err)}, indent=2))
        raise SystemExit(2) from err
    print(json.dumps(Trainer.to_dict(result), indent=2))


def eval_main() -> None:
    parser = argparse.ArgumentParser(prog="pre-eval")
    parser.add_argument("--run-id", default="latest")
    args = parser.parse_args()
    print(json.dumps({"status": "stub", "run_id": args.run_id}, indent=2))


def predict_main() -> None:
    parser = argparse.ArgumentParser(prog="pre-predict")
    parser.add_argument("--run-id", default="latest")
    args = parser.parse_args()
    print(json.dumps({"status": "stub", "run_id": args.run_id}, indent=2))


if __name__ == "__main__":
    train_main()
