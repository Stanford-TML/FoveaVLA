"""Prepare aligned Fast-FoundationStereo depth for the selected model."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("foveavla_single", "foveavla_dual"), required=True)
    default_foveal = Path("data/foveal")
    if not default_foveal.exists() and Path("data/roi").exists():
        default_foveal = Path("data/roi")
    parser.add_argument("--foveal-repack-dir", "--roi-repack-dir", type=Path, default=default_foveal)
    parser.add_argument("--output-dir", type=Path, default=Path("data/depth"))
    parser.add_argument("--zed-settings", type=Path, required=True)
    parser.add_argument("--fast-repo", type=Path, default=Path(".runtime/Fast-FoundationStereo"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    foveal = args.foveal_repack_dir.resolve()
    output = args.output_dir.resolve()
    repo = args.fast_repo.resolve()
    checkpoint = (args.checkpoint or repo / "weights/23-36-37/model_best_bp2_serialize.pth").resolve()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    for path in (foveal / "_SUCCESS", foveal / "rlds_sample_ranges.json", args.zed_settings, checkpoint):
        if not path.exists():
            parser.error(f"Required input does not exist: {path}")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(repo), str(root / "src"), environment.get("PYTHONPATH", "")))

    def run(name, *arguments):
        subprocess.run(
            [sys.executable, str(root / "scripts" / name), *(str(a) for a in arguments)], check=True, env=environment
        )

    filters = ["--input", foveal / "rlds_sample_ranges.json"]
    streams = [("left", "exterior_image_1_left")]
    if args.model == "foveavla_dual":
        streams.append(("right", "exterior_image_2_left"))
    for view, stream in streams:
        manifests = output / f"{view}_manifests"
        audit = output / f"{view}_alignment"
        sidecar = output / view
        plan_path = sidecar / "plan.json"
        if plan_path.is_file():
            plan = json.loads(plan_path.read_text())
            for field, expected in (("manifest_root", manifests), ("checkpoint", checkpoint), ("fast_repo", repo)):
                if plan.get(field) != str(expected):
                    parser.error(f"Existing depth plan has a different {field}: {sidecar}")
        if not (manifests / "_SUCCESS").is_file():
            run(
                "prepare_droid_stereo_geometry_production.py",
                "--foveal-repack-dir",
                foveal,
                "--output-dir",
                manifests,
                "--exterior-view",
                stream,
                "--shards",
                1,
            )
        audit.mkdir(parents=True, exist_ok=True)
        for manifest in sorted((manifests / "manifests").glob("shard-*.npz")):
            run(
                "extract_droid_stereo_alignment_audit.py",
                "--manifest",
                manifest,
                "--output-dir",
                audit / "snapshots",
                "--report",
                audit / "reports" / f"{manifest.stem}.json",
                "--zed-settings",
                args.zed_settings.resolve(),
                "--resume",
            )
        run("validate_droid_stereo_alignment_audit.py", "--manifest-root", manifests, "--audit-root", audit)
        if not (sidecar / "plan.json").is_file():
            run(
                "prepare_droid_fast_foundation_sidecar.py",
                "initialize",
                "--output-dir",
                sidecar,
                "--manifest-root",
                manifests,
                "--foveal-repack-dir",
                foveal,
                "--alignment-audit-root",
                audit,
                "--checkpoint",
                checkpoint,
                "--fast-repo",
                repo,
                "--worker-count",
                1,
                "--batch-size",
                args.batch_size,
            )
        if not (sidecar / "_SUCCESS").is_file():
            run(
                "extract_droid_fast_foundation_depth.py",
                "--plan",
                plan_path,
                "--worker-index",
                0,
                "--zed-settings",
                args.zed_settings.resolve(),
            )
            run("prepare_droid_fast_foundation_sidecar.py", "finalize", "--output-dir", sidecar)
        sample_filter = output / f"{view}_rlds_sample_ranges.json"
        run(
            "build_droid_pointmap_rlds_filter.py",
            "--manifest-root",
            manifests,
            "--episodes",
            sidecar / "episodes.parquet",
            "--base-filter",
            foveal / "rlds_sample_ranges.json",
            "--output",
            sample_filter,
        )
        filters += ["--input", sample_filter]
    run("intersect_droid_rlds_filters.py", *filters, "--output", output / "paired_rlds_sample_ranges.json")
    print(f"Training depth prepared for {args.model}: {output}")


if __name__ == "__main__":
    main()
