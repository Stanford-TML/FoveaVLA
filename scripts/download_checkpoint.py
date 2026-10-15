"""Download a pinned FoveaVLA deployment checkpoint, including its normalization statistics."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

CHECKPOINTS = {
    "foveavla_single": (
        "bingaochen/FoveaVLA-Single",
        "d7d7fdbd6c5abc26106b5128ec8cacb17a531676",
    ),
    "foveavla_dual": (
        "bingaochen/FoveaVLA-Dual",
        "24ca2574c8139c8adad541d2bb4bc476f2c615f8",
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=CHECKPOINTS, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    repo, revision = CHECKPOINTS[args.model]
    checkpoint = args.output_dir / args.model
    snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=checkpoint,
        allow_patterns=["params/**", "assets/**", "_CHECKPOINT_METADATA", "config.json"],
    )
    if not (checkpoint / "params").is_dir() or not (checkpoint / "assets/droid/norm_stats.json").is_file():
        raise RuntimeError(f"Incomplete checkpoint: {checkpoint}")
    print(f"Checkpoint: {checkpoint.resolve()}")
    print(f"Serve: uv run --no-sync scripts/serve.py --model={args.model} --checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
