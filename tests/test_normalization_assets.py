"""Check training and deployment select the intended normalization assets."""

import dataclasses
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from foveavla import transforms
from foveavla.policies import policy_config
from foveavla.shared import normalize
from foveavla.training import config


def write_stats(directory: Path, center: float) -> None:
    normalize.save(
        directory,
        {
            key: normalize.NormStats(
                mean=np.full(8, center),
                std=np.ones(8),
                q01=np.full(8, center - 1),
                q99=np.full(8, center + 1),
            )
            for key in ("state", "actions")
        },
    )


class NormalizationAssetsTest(unittest.TestCase):
    def test_training_uses_base_stats_instead_of_previously_computed_local_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cached = root / "base/assets/droid"
            local = root / "local-assets"
            write_stats(cached, 2)
            write_stats(local / "droid", 99)
            for name in ("foveavla_single", "foveavla_dual"):
                with self.subTest(model=name):
                    cfg = config.get_config(name)
                    expected = cfg.weight_loader.params_path.removesuffix("/params") + "/assets/droid"
                    with mock.patch.object(config._download, "maybe_download", return_value=cached) as download:
                        data = cfg.data.create_base_config(local, cfg.model)
                    download.assert_called_once_with(expected)
                    normalized = transforms.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm)(
                        {"state": np.full(8, 2.0)}
                    )
                    np.testing.assert_allclose(normalized["state"], 0, atol=1e-5)

    def test_local_base_checkpoint_override_loads_its_statistics_for_both_models(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base"
            write_stats(base / "assets/droid", 3)
            code = """
import os
from pathlib import Path
import numpy as np
from foveavla.training import config
base = Path(os.environ['FOVEAVLA_BASE_CHECKPOINT'])
for name in ('foveavla_single', 'foveavla_dual'):
    cfg = config.get_config(name)
    data = cfg.data.create_base_config(base / 'unused-local-assets', cfg.model)
    np.testing.assert_array_equal(data.norm_stats['state'].mean, np.full(8, 3))
    assert cfg.weight_loader.params_path == str(base / 'params')
"""
            subprocess.run(
                [sys.executable, "-c", code],
                env={**os.environ, "FOVEAVLA_BASE_CHECKPOINT": str(base)},
                check=True,
                capture_output=True,
                text=True,
            )

    def test_deployment_uses_checkpoint_stats_without_requesting_base_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            write_stats(checkpoint / "assets/droid", 5)

            def local_only(url, **kwargs):
                self.assertNotIn("://", url)
                return Path(url).resolve()

            for name in ("foveavla_single", "foveavla_dual"):
                with self.subTest(model=name):
                    cfg = config.get_config(name)
                    original_assets = dataclasses.asdict(cfg.data.assets)
                    with (
                        mock.patch.object(policy_config.download, "maybe_download", side_effect=local_only),
                        mock.patch.object(policy_config._model, "restore_params", return_value={}),
                        mock.patch.object(type(cfg.model), "load", return_value=object()),
                        mock.patch.object(config.ModelTransformFactory, "__call__", return_value=transforms.Group()),
                        mock.patch.object(policy_config._policy, "Policy") as policy,
                    ):
                        policy_config.create_trained_policy(cfg, checkpoint)
                    inputs = policy.call_args.kwargs["transforms"]
                    normalizer = next(item for item in inputs if isinstance(item, transforms.Normalize))
                    np.testing.assert_allclose(normalizer({"state": np.full(8, 5.0)})["state"], 0, atol=1e-5)
                    self.assertEqual(dataclasses.asdict(cfg.data.assets), original_assets)


if __name__ == "__main__":
    unittest.main()
