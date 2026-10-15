"""foveal, stereo manifest, and alignment audit fingerprints for metric-depth sidecars."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
from pathlib import Path


# Preserve the serialized identifier so existing depth-sidecar fingerprints still match.
FOVEAL_FINGERPRINT_FORMAT = "droid-roi-metadata-sha256-v1"
MANIFEST_FINGERPRINT_FORMAT = "droid-stereo-production-manifests-sha256-v1"
ALIGNMENT_ATTESTATION_FORMAT = "droid-stereo-alignment-attestation-v2"
ALIGNMENT_VALIDATION_FORMAT = "droid-stereo-alignment-validation-v2"


def foveal_repack_fingerprint(root: str | Path) -> dict[str, str | int]:
    """Fingerprint the interaction box metadata that defines episode joins and crop geometry."""
    root = Path(root)
    index_path = root / "index.json"
    episodes_path = root / "episodes.parquet"
    manifest_dir = root / "manifest"
    if not index_path.is_file() or not episodes_path.is_file() or not manifest_dir.is_dir():
        raise FileNotFoundError(f"foveal repack metadata is incomplete: {root}")
    manifest_paths = sorted(path for path in manifest_dir.rglob("*") if path.is_file())
    if not manifest_paths:
        raise FileNotFoundError(f"foveal repack manifest is empty: {manifest_dir}")

    manifest_hasher = hashlib.sha256()
    for path in manifest_paths:
        relative_path = path.relative_to(manifest_dir).as_posix()
        manifest_hasher.update(relative_path.encode())
        manifest_hasher.update(b"\0")
        manifest_hasher.update(bytes.fromhex(_file_sha256(path)))
    return {
        "format_version": FOVEAL_FINGERPRINT_FORMAT,
        "index_sha256": _file_sha256(index_path),
        "episodes_sha256": _file_sha256(episodes_path),
        "manifest_tree_sha256": manifest_hasher.hexdigest(),
        "manifest_files": len(manifest_paths),
    }


def annotation_keys_sha256(annotation_keys: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for annotation_key in sorted(annotation_keys):
        digest.update(annotation_key.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def production_manifest_fingerprint(root: str | Path) -> dict[str, str | int]:
    root = Path(root)
    index_path = root / "index.json"
    manifest_dir = root / "manifests"
    if not index_path.is_file() or not manifest_dir.is_dir():
        raise FileNotFoundError(f"Production manifest metadata is incomplete: {root}")
    index = json.loads(index_path.read_text())
    expected_files = int(index["shards"])
    manifest_paths = sorted(manifest_dir.glob("shard-*.npz"))
    if len(manifest_paths) != expected_files:
        raise ValueError(
            f"Production manifest file count mismatch in {root}: {len(manifest_paths)} != {expected_files}"
        )

    manifest_hasher = hashlib.sha256()
    for path in manifest_paths:
        manifest_hasher.update(path.name.encode())
        manifest_hasher.update(b"\0")
        manifest_hasher.update(bytes.fromhex(_file_sha256(path)))
    return {
        "format_version": MANIFEST_FINGERPRINT_FORMAT,
        "index_sha256": _file_sha256(index_path),
        "manifest_tree_sha256": manifest_hasher.hexdigest(),
        "manifest_files": len(manifest_paths),
    }


def alignment_audit_attestation(root: str | Path) -> dict[str, object]:
    root = Path(root)
    if not (root / "_SUCCESS").is_file() or (root / "_INCOMPLETE").exists():
        raise RuntimeError(f"Stereo RGB alignment audit is not complete: {root}")
    validation_path = root / "validation.json"
    validation = json.loads(validation_path.read_text())
    summary = validation.get("summary", {})
    if (
        validation.get("format_version") != ALIGNMENT_VALIDATION_FORMAT
        or summary.get("complete") is not True
        or int(summary.get("validation_errors", -1)) != 0
    ):
        raise ValueError(f"Stereo RGB alignment audit did not pass: {validation_path}")
    failures = validation.get("alignment_failures")
    if not isinstance(failures, list) or len(failures) != int(summary.get("alignment_failures", -1)):
        raise ValueError(f"Stereo RGB alignment audit exclusions are invalid: {validation_path}")
    excluded_annotation_keys = sorted(str(row["annotation_key"]) for row in failures)
    if len(excluded_annotation_keys) != len(set(excluded_annotation_keys)):
        raise ValueError(f"Stereo RGB alignment audit exclusions contain duplicates: {validation_path}")
    ambiguities = validation.get("alignment_ambiguities")
    if not isinstance(ambiguities, list) or len(ambiguities) != int(summary.get("alignment_ambiguities", -1)):
        raise ValueError(f"Stereo RGB alignment audit ambiguities are invalid: {validation_path}")
    ambiguous_annotation_keys = sorted(str(row["annotation_key"]) for row in ambiguities)
    if len(ambiguous_annotation_keys) != len(set(ambiguous_annotation_keys)):
        raise ValueError(f"Stereo RGB alignment audit ambiguities contain duplicates: {validation_path}")
    manifest_fingerprint = validation.get("manifest_fingerprint")
    _validate_manifest_fingerprint(manifest_fingerprint, validation_path)
    if production_manifest_fingerprint(validation["manifest_root"]) != manifest_fingerprint:
        raise ValueError(f"Stereo RGB alignment audit manifest fingerprint is stale: {validation_path}")
    annotation_key_hash = validation.get("annotation_keys_sha256")
    _validate_sha256(annotation_key_hash, "annotation_keys_sha256", validation_path)
    return {
        "format_version": ALIGNMENT_ATTESTATION_FORMAT,
        "root": str(root),
        "validation_sha256": _file_sha256(validation_path),
        "manifest_root": str(validation["manifest_root"]),
        "manifest_fingerprint": manifest_fingerprint,
        "episodes": int(summary["episodes"]),
        "alignment_frames": int(summary["alignment_frames"]),
        "annotation_keys_sha256": annotation_key_hash,
        "excluded_annotation_keys": excluded_annotation_keys,
        "ambiguous_annotation_keys": ambiguous_annotation_keys,
        "extractor_source_sha256": str(validation["extractor_source_sha256"]),
        "alignment_source_sha256": str(validation["alignment_source_sha256"]),
        "validator_source_sha256": str(validation["validator_source_sha256"]),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_manifest_fingerprint(fingerprint: object, path: Path) -> None:
    if not isinstance(fingerprint, dict) or fingerprint.get("format_version") != MANIFEST_FINGERPRINT_FORMAT:
        raise ValueError(f"Invalid production manifest fingerprint in {path}")
    _validate_sha256(fingerprint.get("index_sha256"), "manifest index_sha256", path)
    _validate_sha256(fingerprint.get("manifest_tree_sha256"), "manifest_tree_sha256", path)
    if not isinstance(fingerprint.get("manifest_files"), int) or int(fingerprint["manifest_files"]) < 1:
        raise ValueError(f"Invalid manifest_files in {path}")


def _validate_sha256(value: object, name: str, path: Path) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Invalid {name} in {path}")
