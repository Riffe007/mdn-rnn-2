from __future__ import annotations

import argparse
import json

from pre.demo.runner import build_mode_cards, run_all_demos, run_demo


def demo_main() -> None:
    parser = argparse.ArgumentParser(prog="pre-demo")
    parser.add_argument("--mode", default="all")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--list-modes", action="store_true")
    args = parser.parse_args()

    if args.list_modes:
        print(json.dumps(build_mode_cards(), indent=2))
        return

    if args.mode == "all":
        payload = run_all_demos(artifact_root=args.artifact_root)
    else:
        payload = run_demo(mode=args.mode, artifact_root=args.artifact_root)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    demo_main()
