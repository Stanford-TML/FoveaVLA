"""Regression checks for renaming fields without changing released data or inputs."""

import json
from pathlib import Path
import struct
import tempfile
import unittest

import lmdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from foveavla_client import foveation, image_tools

from droid_foveation.parquet_io import read_frame_table
from droid_foveation.repack_direct import pack_value
from foveavla.models.tokenizer import PaligemmaTokenizer
from foveavla.policies.droid_policy import DroidInputs, DroidOutputs
from foveavla.policies.policy import Policy
from foveavla.shared import stereo_geometry_manifest as geometry
from foveavla.training import droid_pointmap_store as pointmaps
from foveavla.training import droid_stereo_geometry_store as fingerprints
from foveavla.training.droid_foveal_dataset import _unpack_value, decode_foveal_sample
from foveavla.transforms import TokenizePrompt


class FoveationCompatibilityTest(unittest.TestCase):
    def test_canonical_fields_win_without_mutating_legacy_input(self):
        data = {"future_interaction_roi": [0, 0, 1, 1], "future_interaction_box": [0.1, 0.2, 0.3, 0.4]}
        normalized = foveation.canonicalize_fields(data)
        self.assertEqual(normalized, {"future_interaction_box": [0.1, 0.2, 0.3, 0.4]})
        self.assertIn("future_interaction_roi", data)

    def test_legacy_single_and_dual_requests_produce_identical_model_inputs(self):
        for dual in (False, True):
            with self.subTest(dual=dual):
                views = ("left_", "right_") if dual else ("",)
                data = {
                    "observation/joint_position": np.zeros(7),
                    "observation/gripper_position": 0.5,
                    "future_interaction_box": np.ones((2, 4) if dual else (4,)) * 0.5,
                    "future_interaction_box_mask": np.ones((2,) if dual else (), dtype=bool),
                }
                for index, view in enumerate(views, 1):
                    data[f"observation/exterior_image_{index}_left"] = np.full((224, 224, 3), index, np.uint8)
                    data[f"observation/{view}foveal_image"] = np.full((224, 224, 3), index + 10, np.uint8)
                    for kind in ("full", "crop"):
                        data[f"observation/metric_depth_{view}{kind}"] = np.ones((224, 224), np.float32)
                        data[f"observation/pointmap_quality_{view}{kind}"] = np.ones((224, 224), np.float32)
                legacy = {foveation.FIELD_ALIASES.get(key, key): value for key, value in data.items()}
                current_result = DroidInputs(dual)(data)
                legacy_result = DroidInputs(dual)(legacy)
                for key in ("state", "future_interaction_box", "future_interaction_box_mask"):
                    np.testing.assert_array_equal(current_result[key], legacy_result[key])
                for kind in ("image", "depth", "depth_quality"):
                    self.assertEqual(current_result[kind].keys(), legacy_result[kind].keys())
                    for slot in current_result[kind]:
                        np.testing.assert_array_equal(current_result[kind][slot], legacy_result[kind][slot])

    def test_training_targets_keep_the_trained_tokenizer_prefix(self):
        class Encoder:
            def encode(self, text, **kwargs):
                self.last_text = text
                return [3, 4]

        tokenizer = PaligemmaTokenizer.__new__(PaligemmaTokenizer)
        tokenizer._max_len = 20
        tokenizer._tokenizer = Encoder()
        data = {
            "prompt": "pick up the object",
            "state": np.zeros(8),
            "actions": np.zeros((16, 8)),
            "future_interaction_box": np.array([0.1, 0.2, 0.6, 0.7], dtype=np.float32),
            "future_interaction_box_mask": True,
        }
        for count in (1, 2):
            with self.subTest(count=count):
                if count == 2:
                    data["future_interaction_box"] = np.stack([data["future_interaction_box"]] * 2)
                    data["future_interaction_box_mask"] = np.ones(2, dtype=bool)
                legacy = {foveation.FIELD_ALIASES.get(key, key): value for key, value in data.items()}
                transform = TokenizePrompt(tokenizer, future_bbox_count=count)
                current = transform(dict(data))
                old = transform(dict(legacy))
                for key in ("tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask"):
                    np.testing.assert_array_equal(current[key], old[key])
                self.assertEqual(tokenizer._tokenizer.last_text, "Future interaction ROI: ")

    def test_legacy_parquet_fields_remain_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            for legacy in (False, True):
                names = (
                    ("interaction_roi", "future_interaction_roi")
                    if legacy
                    else ("interaction_box", "future_interaction_box")
                )
                path = Path(directory) / f"frames-{legacy}.parquet"
                pq.write_table(pa.table({names[0]: [[0.1, 0.2, 0.3, 0.4]], names[1]: [[0.2, 0.3, 0.4, 0.5]]}), path)
                table = read_frame_table(path, ["interaction_box", "future_interaction_box"])
                self.assertEqual(table.column_names, ["interaction_box", "future_interaction_box"])
                self.assertEqual(table["future_interaction_box"].to_pylist(), [[0.2, 0.3, 0.4, 0.5]])

    def test_geometry_manifests_accept_both_path_field_names(self):
        episode = geometry.GeometryManifestEpisode(
            "annotation",
            "serial",
            "video.svo",
            "video.mp4",
            np.array([0], np.int32),
            np.array([2], np.int8),
            np.array([[10, 20, 30, 40]], np.int32),
        )
        manifest = geometry.GeometryManifest(
            geometry.MANIFEST_FORMAT_VERSION, geometry.EXTERIOR_VIEW, "/existing/crops", 0, 1, (episode,)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.npz"
            geometry.write_geometry_manifest(path, manifest)
            self.assertEqual(geometry.load_geometry_manifest(path).foveal_repack_dir, "/existing/crops")
            with np.load(path) as payload:
                legacy = {key: payload[key] for key in payload.files}
            legacy["roi_repack_dir"] = legacy.pop("foveal_repack_dir")
            np.savez(path, **legacy)
            loaded = geometry.load_geometry_manifest(path)
            self.assertEqual(loaded.foveal_repack_dir, "/existing/crops")
            np.testing.assert_array_equal(loaded.episodes[0].crop_bbox, episode.crop_bbox)
            json_path = Path(directory) / "manifest.json"
            payload = {
                "format_version": geometry.MANIFEST_FORMAT_VERSION,
                "exterior_view": geometry.EXTERIOR_VIEW,
                "roi_repack_dir": "/existing/crops",
                "episodes": [
                    {
                        "annotation_key": "annotation",
                        "camera_serial": "serial",
                        "svo_path": "video.svo",
                        "video_path": "video.mp4",
                        "samples": [{"frame_idx": 0, "crop_mode": 2, "crop_bbox": [10, 20, 30, 40]}],
                    }
                ],
            }
            json_path.write_text(json.dumps(payload))
            self.assertEqual(geometry.load_geometry_manifest(json_path).foveal_repack_dir, "/existing/crops")

    def test_released_lmdb_binary_layout_still_decodes(self):
        import cv2

        image = np.full((224, 224, 3), [10, 20, 30], dtype=np.uint8)
        ok, png = cv2.imencode(".png", image)
        self.assertTrue(ok)
        box, crop = (0.1, 0.2, 0.6, 0.7), (10, 20, 234, 244)
        legacy_payload = struct.pack("<4fB3x4iI", *box, 2, *crop, len(png)) + png.tobytes()
        self.assertEqual(pack_value(box, 2, crop, png.tobytes()), legacy_payload)
        sample = _unpack_value("annotation--0", legacy_payload)
        decoded = decode_foveal_sample(sample, output_size=224)
        np.testing.assert_allclose(decoded.future_interaction_box, box)
        np.testing.assert_array_equal(decoded.context_image, image[..., ::-1])

    def test_depth_sources_accept_legacy_fingerprints_without_rewriting_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            crops = root / "crops"
            (crops / "manifest").mkdir(parents=True)
            (crops / "index.json").write_text("{}")
            rows = pa.table(
                {"annotation_key": ["annotation"], "ori_path": ["episode"], "view": [geometry.EXTERIOR_VIEW]}
            )
            pq.write_table(rows, crops / "episodes.parquet")
            pq.write_table(rows, crops / "manifest/part-00000.parquet")
            fingerprint = fingerprints.foveal_repack_fingerprint(crops)
            self.assertEqual(fingerprint["format_version"], "droid-roi-metadata-sha256-v1")
            depth = root / "depth"
            (depth / "pointmaps").mkdir(parents=True)
            (depth / "_SUCCESS").touch()
            env = lmdb.open(str(depth / "pointmaps/shard-000.lmdb"))
            env.close()
            pq.write_table(rows, depth / "episodes.parquet")
            index = {
                "format_version": pointmaps.FORMAT_VERSION,
                "exterior_view": geometry.EXTERIOR_VIEW,
                "store_shards": 1,
                "pointmap": {
                    "shape": list(pointmaps.POINTMAP_SHAPE),
                    "model_dtype": "float32",
                    "frame": "zed_left_camera",
                    "normalization_scale_m": 1,
                    "storage": "aligned_metric_depth_plus_camera_model",
                    "depth_dtype": pointmaps.DEPTH_DTYPE.str,
                },
                "quality": {"shape": list(pointmaps.QUALITY_SHAPE), "storage_dtype": pointmaps.QUALITY_DTYPE.str},
                "value": {"compression": "zlib", "raw_bytes": pointmaps.RAW_VALUE_BYTES},
                "depth_source": {
                    "sensor": "ZED stereo SVO rectified pair",
                    "estimator": "Fast-FoundationStereo",
                    "alignment": "left",
                    "metric_conversion": "fx_times_baseline_over_disparity",
                    "range_m": [0.2, 3.0],
                    "checkpoint_sha256": "0" * 64,
                },
            }
            for legacy in (False, True):
                prefix = "roi" if legacy else "foveal"
                index["source"] = {
                    f"{prefix}_repack_dir": str(crops),
                    f"{prefix}_repack_fingerprint": fingerprint,
                    "alignment_audit": {},
                    "selected_episodes": 1,
                    "selected_annotation_keys_sha256": fingerprints.annotation_keys_sha256(["annotation"]),
                }
                (depth / "index.json").write_text(json.dumps(index))
                store = pointmaps.PointmapStore(depth)
                try:
                    self.assertEqual(store.foveal_repack_dir, str(crops))
                    store.validate_foveal_repack(crops)
                finally:
                    store.close()

    def test_sync_and_async_predictions_expose_released_aliases(self):
        policy = Policy.__new__(Policy)
        box = np.array([[0.1, 0.2, 0.6, 0.7]])
        valid = np.array([True])
        policy._prepare_inference = lambda *args: ({"state": np.zeros((1, 8))}, None, None, {})
        policy._sample_actions_with_future_bbox = lambda *args, **kwargs: (np.zeros((1, 16, 32)), box, valid)
        policy._sample_future_bbox_from_prepared_prefix = lambda *args: (box, valid)
        policy._output_transform = DroidOutputs()
        for output in (policy.infer({}), policy.infer_future_bbox(type("Pending", (), {"prefix": None})())):
            np.testing.assert_array_equal(output["future_interaction_box"], output["future_interaction_roi"])
            np.testing.assert_array_equal(
                output["future_interaction_box_valid"], output["future_interaction_roi_valid"]
            )

    def test_legacy_client_crop_keyword_arguments(self):
        image = np.arange(80 * 120 * 3, dtype=np.uint8).reshape(80, 120, 3)
        box = np.array([0.2, 0.3, 0.5, 0.6])
        crop, bounds = image_tools.extract_foveal_crop_direct_with_bbox(image, box)
        old_crop, old_bounds = image_tools.extract_roi_context_direct_with_bbox(image, roi=box)
        np.testing.assert_array_equal(crop, old_crop)
        self.assertEqual(bounds, old_bounds)


if __name__ == "__main__":
    unittest.main()
