"""Runtime wrapper for the exact Fast-FoundationStereo training estimator."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
import types

import numpy as np


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FastFoundationStereoRuntime:
    """Load Fast-FoundationStereo and infer one left-reference disparity map."""

    def __init__(
        self,
        *,
        repository: str | Path,
        checkpoint: str | Path,
        repository_commit: str,
        checkpoint_sha256: str,
        input_resolution: tuple[int, int],
        max_disparity: int,
        valid_iters: int,
        low_memory: bool,
    ) -> None:
        repository = Path(repository).resolve()
        checkpoint = Path(checkpoint).resolve()
        if not repository.is_dir() or not checkpoint.is_file():
            raise FileNotFoundError("Fast-FoundationStereo repository or checkpoint is missing")
        actual_commit = subprocess.run(
            ("git", "-C", str(repository), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual_commit != repository_commit:
            raise ValueError(f"Fast-FoundationStereo commit mismatch: {actual_commit} != {repository_commit}")
        tracked_changes = subprocess.run(
            ("git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if tracked_changes:
            raise ValueError("Fast-FoundationStereo repository has tracked changes")
        if _sha256_file(checkpoint) != checkpoint_sha256:
            raise ValueError("Fast-FoundationStereo checkpoint hash mismatch")

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Fast-FoundationStereo online preprocessing requires a CUDA GPU")
        if str(repository) not in sys.path:
            sys.path.insert(0, str(repository))
        sys.modules.setdefault("open3d", types.ModuleType("open3d"))
        from core.utils.utils import InputPadder
        from Utils import AMP_DTYPE

        model = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.args.valid_iters = valid_iters
        model.args.max_disp = max_disparity
        if "normalize" not in model.args:
            model.args.normalize = True

        self._torch = torch
        self._input_padder = InputPadder
        self._amp_dtype = AMP_DTYPE
        self._model = model.cuda().eval()
        self.input_resolution = input_resolution
        self.valid_iters = valid_iters
        self.low_memory = low_memory

    def infer(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
        width, height = self.input_resolution
        left_rgb = np.asarray(left_rgb)
        right_rgb = np.asarray(right_rgb)
        expected_shape = (height, width, 3)
        if left_rgb.shape != expected_shape or right_rgb.shape != expected_shape:
            raise ValueError(f"Expected rectified stereo images shaped {expected_shape}")
        if left_rgb.dtype != np.uint8 or right_rgb.dtype != np.uint8:
            raise ValueError("Fast-FoundationStereo inputs must be uint8 RGB")

        left = self._torch.as_tensor(left_rgb[None], device="cuda").float().permute(0, 3, 1, 2)
        right = self._torch.as_tensor(right_rgb[None], device="cuda").float().permute(0, 3, 1, 2)
        padder = self._input_padder(left.shape, divis_by=32, force_square=False)
        left, right = padder.pad(left, right)
        with self._torch.inference_mode(), self._torch.amp.autocast("cuda", enabled=True, dtype=self._amp_dtype):
            disparity = self._model.forward(
                left,
                right,
                iters=self.valid_iters,
                test_mode=True,
                low_memory=self.low_memory,
                optimize_build_volume="pytorch1",
            )
        disparity = padder.unpad(disparity.float())[0, 0].cpu().numpy()
        if disparity.shape != (height, width) or not np.all(np.isfinite(disparity)):
            raise RuntimeError(f"Fast-FoundationStereo returned invalid disparity shape/content: {disparity.shape}")
        return disparity
