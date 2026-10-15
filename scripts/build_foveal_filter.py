"""Join successfully rendered foveal frames to DROID's non-idle RLDS frame filter."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from droid_foveation.repack import _intersect_ranges
from droid_foveation.repack import _rlds_episode_key


def build_filter(foveal: Path, base_filter: dict, model: str) -> dict:
    view_ranges = {}
    for part in sorted((foveal / "manifest").glob("part-*.parquet")):
        for batch in pq.ParquetFile(part).iter_batches(columns=["ori_path", "view", "frame_idx"]):
            for path, view, frame in zip(*(column.to_pylist() for column in batch.columns), strict=True):
                key = _rlds_episode_key(path)
                if key not in base_filter:
                    continue
                ranges = view_ranges.setdefault(key, {}).setdefault(view, [])
                if ranges and frame < ranges[-1][1]:
                    raise ValueError(f"Duplicate or unordered frame: {key} {view} {frame}")
                if ranges and frame == ranges[-1][1]:
                    ranges[-1][1] = frame + 1
                else:
                    ranges.append([frame, frame + 1])
    result = {}
    for key, views in view_ranges.items():
        ranges = views.get("exterior_image_1_left", [])
        if model == "foveavla_dual":
            ranges = _intersect_ranges(ranges, views.get("exterior_image_2_left", []))
        ranges = _intersect_ranges(ranges, base_filter[key])
        if ranges:
            result[key] = ranges
    if not result:
        raise ValueError("No rendered foveal frames overlap the RLDS filter")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foveal-repack-dir", "--roi-repack-dir", type=Path, required=True)
    parser.add_argument("--base-filter", type=Path, required=True)
    parser.add_argument("--model", choices=("foveavla_single", "foveavla_dual"), required=True)
    args = parser.parse_args()
    if not (args.foveal_repack_dir / "_SUCCESS").is_file():
        parser.error("foveal repack is incomplete")
    result = build_filter(args.foveal_repack_dir, json.loads(args.base_filter.read_text()), args.model)
    target = args.foveal_repack_dir / "rlds_sample_ranges.json"
    if target.exists():
        parser.error(f"Output already exists: {target}")
    target.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"RLDS filter: {len(result)} episodes, {sum(b - a for r in result.values() for a, b in r)} frames")


if __name__ == "__main__":
    main()
