"""Download both exterior cameras' MP4/SVO files from public DROID 1.0.1."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import urllib.error
import urllib.parse
import urllib.request

from droid_foveation.sources import read_episode_metadata
from tqdm import tqdm


def exterior_files(raw_root: Path) -> tuple[list[Path], dict[str, str]]:
    metadata_by_episode = {}
    for path in sorted(raw_root.rglob("metadata*.json")):
        metadata_by_episode.setdefault(path.parent, []).append(path)
    files = []
    skipped = {}
    for episode, metadata_paths in metadata_by_episode.items():
        reason = None
        _, metadata = read_episode_metadata(metadata_paths)
        if metadata is None:
            reason = "ambiguous_metadata"
        elif not (episode / "trajectory.h5").is_file():
            reason = "missing_trajectory"
        else:
            serials = [str(metadata.get(f"ext{view}_cam_serial", "")) for view in (1, 2)]
            wrist = str(metadata.get("wrist_cam_serial", ""))
            if not all(serial.isdigit() for serial in serials) or len(set(serials)) != 2 or wrist in serials:
                reason = "invalid_exterior_camera_roles"
        if reason is not None:
            skipped[str(episode.relative_to(raw_root))] = reason
            continue
        for serial in serials:
            for kind, extension in (("MP4", "mp4"), ("SVO", "svo")):
                files.append(episode / "recordings" / kind / f"{serial}.{extension}")
    return files, skipped


def download_file(remote: str, local: Path) -> str:
    try:
        with urllib.request.urlopen(urllib.request.Request(remote, method="HEAD"), timeout=60) as response:
            size = int(response.headers["Content-Length"])
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return "missing"
        raise
    if local.is_file() and local.stat().st_size == size:
        return "existing"
    local.parent.mkdir(parents=True, exist_ok=True)
    temporary = local.with_name(local.name + ".partial")
    with urllib.request.urlopen(remote, timeout=60) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    if temporary.stat().st_size != size:
        raise OSError(f"Incomplete download: {remote}")
    temporary.replace(local)
    return "downloaded"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-droid-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    raw = args.raw_droid_root.resolve()
    if not raw.is_dir() or args.workers < 1:
        parser.error("raw-droid-root must exist and workers must be positive")
    files, skipped = exterior_files(raw)
    if not files:
        parser.error("No episodes with trajectory.h5 and consistent exterior-camera metadata")
    remote_root = "https://storage.googleapis.com/gresearch/robotics/droid_raw/1.0.1"

    def download(path):
        relative = urllib.parse.quote(path.relative_to(raw).as_posix(), safe="/")
        return download_file(f"{remote_root}/{relative}", path)

    counts = {"downloaded": 0, "existing": 0, "missing": 0}
    missing = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for path, status in zip(files, tqdm(executor.map(download, files), total=len(files), unit="file"), strict=True):
            counts[status] += 1
            if status == "missing":
                missing.append(path.relative_to(raw).as_posix())
    report = {"files": counts, "missing_public_files": missing, "skipped_episodes": skipped}
    (raw / "exterior_download_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(counts, indent=2))
    print(f"Skipped episodes: {len(skipped)}; report: {raw / 'exterior_download_report.json'}")


if __name__ == "__main__":
    main()
