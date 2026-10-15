#!/usr/bin/env python3
"""Intersect exact DROID RLDS range filters without expanding frame indices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FORMAT_VERSION = "droid-rlds-filter-intersection-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_ranges(ranges: object, *, key: str) -> list[list[int]]:
    if not isinstance(ranges, list) or not ranges:
        raise ValueError(f"Filter has no ranges for {key}")
    previous_end = -1
    for retained_range in ranges:
        if (
            not isinstance(retained_range, list)
            or len(retained_range) != 2
            or any(type(value) is not int for value in retained_range)
        ):
            raise ValueError(f"Invalid range for {key}: {retained_range}")
        start, end = retained_range
        if start < 0 or start >= end or start < previous_end:
            raise ValueError(f"Overlapping or invalid range for {key}: {retained_range}")
        previous_end = end
    return ranges


def intersect_ranges(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    result = []
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        if start < end:
            result.append([start, end])
        if left[left_index][1] < right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return result


def main() -> None:
    args = parse_args()
    if len(args.input) < 2:
        raise ValueError("At least two --input filters are required")
    resolved_inputs = [path.resolve() for path in args.input]
    if len(resolved_inputs) != len(set(resolved_inputs)):
        raise ValueError("Input filters must be distinct")

    filters = []
    for path in args.input:
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or not value:
            raise ValueError(f"Input filter is empty or invalid: {path}")
        filters.append(value)

    common_keys = set(filters[0]).intersection(*(set(value) for value in filters[1:]))
    result = {}
    samples = 0
    for key in sorted(common_keys):
        ranges = validate_ranges(filters[0][key], key=key)
        for value in filters[1:]:
            ranges = intersect_ranges(ranges, validate_ranges(value[key], key=key))
            if not ranges:
                break
        if ranges:
            result[key] = ranges
            samples += sum(end - start for start, end in ranges)
    if not result:
        raise RuntimeError("Filter intersection is empty")

    output_bytes = (json.dumps(result, separators=(",", ":")) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_bytes() != output_bytes:
        raise FileExistsError(f"Output exists with different content: {args.output}")
    args.output.write_bytes(output_bytes)
    components = []
    for path in args.input:
        summary_path = path.with_suffix(".summary.json")
        components.append(
            {
                "filter": str(path.resolve()),
                "filter_sha256": sha256_file(path),
                "summary": str(summary_path.resolve()) if summary_path.is_file() else None,
                "summary_sha256": sha256_file(summary_path) if summary_path.is_file() else None,
            }
        )
    summary = {
        "format_version": FORMAT_VERSION,
        "episodes": len(result),
        "samples": samples,
        "ranges": sum(len(ranges) for ranges in result.values()),
        "output": str(args.output.resolve()),
        "filter_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "components": components,
        "builder_source_sha256": sha256_file(Path(__file__)),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
