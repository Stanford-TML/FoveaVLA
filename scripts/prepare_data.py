"""Generate future interaction boxes and direct training crops."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("foveavla_single", "foveavla_dual"), required=True)
    parser.add_argument("--qsheet", type=Path, required=True)
    parser.add_argument("--annotation-lmdb", type=Path, required=True)
    parser.add_argument("--raw-droid-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--exterior-roles", type=Path)
    parser.add_argument("--keys-file", type=Path, help="RoboInter annotation keys for the downloaded DROID subset")
    parser.add_argument("--rlds-filter", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--store-shards", type=int, default=16)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    sidecar = output / "annotations"
    repack = output / "foveal"
    if args.workers < 1 or args.store_shards < 1:
        parser.error("workers and store-shards must be positive")
    for path in (args.qsheet, args.annotation_lmdb, args.raw_droid_root, args.rlds_filter):
        if not path.exists():
            parser.error(f"Input does not exist: {path}")
    if args.keys_file is not None and not args.keys_file.is_file():
        parser.error(f"Keys file does not exist: {args.keys_file}")
    if sidecar.exists() or repack.exists() or (output / "roi").exists():
        parser.error("Use a fresh output directory; annotations/ or foveal/ already exists")
    annotate = [
        sys.executable,
        "-m",
        "droid_foveation.annotate",
        "--qsheet",
        str(args.qsheet.resolve()),
        "--annotation-lmdb",
        str(args.annotation_lmdb.resolve()),
        "--raw-droid-root",
        str(args.raw_droid_root.resolve()),
        "--role-policy",
        str(root / "configs/role_policy_v1.yaml"),
        "--future-offset",
        "8",
        "--output-dir",
        str(sidecar),
    ]
    if args.exterior_roles is not None:
        annotate += ["--exterior-roles", str(args.exterior_roles.resolve())]
    if args.keys_file is not None:
        annotate += ["--keys-file", str(args.keys_file.resolve())]
    subprocess.run(annotate, check=True)
    command = [
        sys.executable,
        "-m",
        "droid_foveation.repack_direct",
        "--sidecar",
        str(sidecar),
        "--output-dir",
        str(repack),
        "--output-size",
        "224",
        "--context-scale",
        "1.5",
        "--full-image-probability",
        "0.10",
        "--jitter-probability",
        "0.45",
        "--workers",
        str(args.workers),
        "--store-shards",
        str(args.store_shards),
    ]
    command += ["--paired-views"] if args.model == "foveavla_dual" else ["--view", "exterior_image_1_left"]
    subprocess.run(command, check=True)
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/build_foveal_filter.py"),
            "--foveal-repack-dir",
            str(repack),
            "--base-filter",
            str(args.rlds_filter.resolve()),
            "--model",
            args.model,
        ],
        check=True,
    )
    print(f"Training foveal data: {repack}")
    print(f"RLDS sample filter: {repack / 'rlds_sample_ranges.json'}")


if __name__ == "__main__":
    main()
