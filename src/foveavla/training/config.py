"""FoveaVLA-Single and FoveaVLA-Dual training configurations."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import os
import pathlib
from typing import Any, Protocol, TypeAlias
import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro
import foveavla.models.model as _model
import foveavla.models.foveavla_config as foveavla_config
import foveavla.models.tokenizer as _tokenizer
import foveavla.policies.droid_policy as droid_policy
import foveavla.shared.download as _download
import foveavla.shared.normalize as _normalize
import foveavla.training.droid_rlds_dataset as droid_rlds_dataset
import foveavla.training.optimizer as _optimizer
import foveavla.training.weight_loaders as weight_loaders
import foveavla.transforms as _transforms


Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Normalization statistics copied into each checkpoint under assets/<asset_id>."""

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # Dataset identifier used for normalization assets.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Optional completed foveal crop repack joined to DROID RLDS samples on the host.
    foveal_repack_dir: str | None = None
    foveal_decode_workers: int = 8
    foveal_excluded_annotation_keys: tuple[str, ...] = ()
    pointmap_dir: str | None = None
    pointmap_aux_dirs: tuple[str, ...] = ()
    dual_exterior_views: bool = False
    rlds_shuffle_buffer_size: int = 250_000
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Resize aligned inputs and tokenize the task, state, and future interaction box targets."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        return _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(self.default_prompt),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    future_bbox_count=model_config.future_bbox_count,
                ),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ],
        )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # Dataset identifier used for normalization assets.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=True,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    foveal_repack_dir: str | None = None
    foveal_decode_workers: int = 8
    foveal_excluded_annotation_keys: tuple[str, ...] = ("72929_exterior_image_1_left",)
    depth_dir: str | None = None
    depth_aux_dirs: tuple[str, ...] = ()
    dual_exterior_views: bool = False
    rlds_shuffle_buffer_size: int = 250_000

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if self.foveal_repack_dir is None or self.depth_dir is None:
            raise ValueError("FoveaVLA requires foveal and metric-depth sidecars, without a wrist image")
        expected_bbox_count = 2 if self.dual_exterior_views else 1
        expected_image_keys = ("base_0_rgb", "left_wrist_0_rgb")
        if self.dual_exterior_views:
            expected_image_keys += ("base_1_rgb", "right_wrist_0_rgb")
        if model_config.future_bbox_count != expected_bbox_count or model_config.image_keys != expected_image_keys:
            raise ValueError("Model image slots and future boxes must match the selected exterior views")
        if model_config.depth_image_keys != expected_image_keys:
            raise ValueError("Each global/foveal RGB stream requires its matching metric depth")
        if len(self.depth_aux_dirs) != int(self.dual_exterior_views):
            raise ValueError("Dual requires a second depth sidecar; Single requires only the left sidecar")
        if self.rlds_data_dir is None:
            raise ValueError("Set the RLDS root for training")

        mapping = {
            "observation/joint_position": "observation/joint_position",
            "observation/gripper_position": "observation/gripper_position",
            "actions": "actions",
            "prompt": "prompt",
            "future_interaction_box": "future_interaction_box",
            "future_interaction_box_mask": "future_interaction_box_mask",
        }
        if self.dual_exterior_views:
            mapping.update(
                {
                    "observation/exterior_image_1_left": "observation/left_image",
                    "observation/exterior_image_2_left": "observation/right_image",
                    "observation/left_foveal_image": "observation/left_foveal_image",
                    "observation/right_foveal_image": "observation/right_foveal_image",
                }
            )
        else:
            mapping.update(
                {
                    "observation/exterior_image_1_left": "observation/image",
                    "observation/foveal_image": "observation/foveal_image",
                }
            )
        suffixes = (
            ("left_full", "left_crop", "right_full", "right_crop") if self.dual_exterior_views else ("full", "crop")
        )
        for suffix in suffixes:
            for field in ("metric_depth", "pointmap_quality"):
                key = f"observation/{field}_{suffix}"
                mapping[key] = key
        # Historical sidecar quality field names are retained for the prepared data.
        data_transforms = _transforms.Group(
            inputs=[
                droid_policy.DroidInputs(dual_exterior_views=self.dual_exterior_views),
                _transforms.DeltaActions(_transforms.make_bool_mask(7, -1)),
            ],
            outputs=[
                _transforms.AbsoluteActions(_transforms.make_bool_mask(7, -1)),
                droid_policy.DroidOutputs(),
            ],
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=_transforms.Group(inputs=[_transforms.RepackTransform(mapping)]),
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            rlds_data_dir=self.rlds_data_dir,
            foveal_repack_dir=self.foveal_repack_dir,
            foveal_decode_workers=self.foveal_decode_workers,
            foveal_excluded_annotation_keys=self.foveal_excluded_annotation_keys,
            pointmap_dir=self.depth_dir,
            pointmap_aux_dirs=self.depth_aux_dirs,
            dual_exterior_views=self.dual_exterior_views,
            rlds_shuffle_buffer_size=self.rlds_shuffle_buffer_size,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "FoveaVLA"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., FoveaVLAConfig) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=foveavla_config.FoveaVLAConfig)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=RLDSDroidDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


_DATA_ROOT = pathlib.Path(os.environ.get("FOVEAVLA_DATA_ROOT", "./data"))
_RLDS_ROOT = os.environ.get("FOVEAVLA_RLDS_DIR", str(_DATA_ROOT / "rlds"))
_DEFAULT_FOVEAL_ROOT = _DATA_ROOT / "foveal"
if not _DEFAULT_FOVEAL_ROOT.exists() and (_DATA_ROOT / "roi").exists():
    _DEFAULT_FOVEAL_ROOT = _DATA_ROOT / "roi"
_FOVEAL_ROOT = os.environ.get(
    "FOVEAVLA_FOVEAL_DIR", os.environ.get("FOVEAVLA_ROI_DIR", str(_DEFAULT_FOVEAL_ROOT))
)
_DEPTH_LEFT = os.environ.get("FOVEAVLA_DEPTH_LEFT_DIR", str(_DATA_ROOT / "depth/left"))
_DEPTH_RIGHT = os.environ.get("FOVEAVLA_DEPTH_RIGHT_DIR", str(_DATA_ROOT / "depth/right"))
_DEPTH_FILTER = os.environ.get("FOVEAVLA_DEPTH_FILTER", str(_DATA_ROOT / "depth/paired_rlds_sample_ranges.json"))
_BASE_CHECKPOINT = os.environ.get("FOVEAVLA_BASE_CHECKPOINT", "gs://openpi-assets/checkpoints/pi05_base")


_single_config = TrainConfig(
    name="foveavla_single",
    model=foveavla_config.FoveaVLAConfig(
        action_dim=32,
        action_horizon=16,
        # Keep checkpoint slot names; the second slot contains the foveal crop, with no wrist camera.
        image_keys=("base_0_rgb", "left_wrist_0_rgb"),
        depth_image_keys=("base_0_rgb", "left_wrist_0_rgb"),
        depth_view_dropout_prob=0.1,
        bbox_loss_weight=0.1,
    ),
    data=RLDSDroidDataConfig(
        repo_id="droid",
        rlds_data_dir=_RLDS_ROOT,
        foveal_repack_dir=_FOVEAL_ROOT,
        foveal_decode_workers=32,
        depth_dir=_DEPTH_LEFT,
        datasets=(
            droid_rlds_dataset.RLDSDataset(name="droid", version="1.0.1", weight=1.0, filter_dict_path=_DEPTH_FILTER),
        ),
        assets=AssetsConfig(assets_dir=_BASE_CHECKPOINT + "/assets", asset_id="droid"),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(_BASE_CHECKPOINT + "/params"),
    lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1000, peak_lr=5e-05, decay_steps=1000000, decay_lr=5e-05),
    num_train_steps=100000,
    batch_size=256,
    log_interval=100,
    save_interval=5000,
    keep_period=10000,
    fsdp_devices=8,
    policy_metadata={
        "model_name": "FoveaVLA-Single",
        "action_space": "joint_position",
        "requires_wrist_camera": False,
        "droid_exterior_views": ("physical_left",),
        "future_bbox_horizon": 8,
        "future_bbox_views": ("physical_left",),
        "foveal_context_scale": 1.5,
        "foveal_context_format": "direct_v2.2.0",
        "requires_metric_depth": True,
        "metric_depth_shape": (224, 224),
        "metric_depth_views": ("full", "foveal"),
        "metric_depth_quantity": "optical_axis_z",
        "metric_depth_reference": "zed_left_camera_optical_center",
        "metric_depth_unit": "meter",
        "metric_depth_range_m": (0.2, 3.0),
        "metric_depth_encoding": "fixed_log_turbo_polynomial_v1",
        "metric_depth_colormap_direction": "near_warm_far_cool",
        "metric_depth_invalid_encoding": "black_with_siglip_patch_mask",
        "metric_depth_quality_min": 0.1,
        "metric_depth_patch_min_valid_fraction": 0.25,
        "metric_depth_encoder": "shared-SigLIP-So400m/14",
        "metric_depth_fusion": "separate_prefix_token_stream",
        "metric_depth_updates_shared_siglip": True,
        "metric_depth_color_augmentation": False,
        "metric_depth_view_dropout_probability": 0.1,
        "metric_depth_source": "Fast-FoundationStereo-a290ba04",
        "metric_depth_input_contract": "left_aligned_fast_foundation_metric_depth_and_binary_validity",
        "pointmap_frame": "zed_left_camera",
        "requires_camera_to_robot_calibration": False,
        "metric_depth_online_offline_parity": True,
        "requires_fast_foundation_stereo_pair": True,
        # Versioned wire identifier retained for released clients.
        "fast_foundation_online_input_contract": "rectified_rgb_pair_fx_baseline_and_direct_roi_crop_v1",
        "fast_foundation_sensor": "ZED stereo SVO rectified pair",
        "fast_foundation_zed_sdk_version": "5.1.2",
        "fast_foundation_repository_commit": "a290ba04c1b3ad1ec41a33974a157b2917b624d4",
        "fast_foundation_postprocess_source_sha256": "e28bc51ebe22e64ef8d9cf7920f2a81ba6f3599d74596156d3f11b03fb4bec7a",
        "fast_foundation_checkpoint_sha256": "7aee85948373da62b0503c2542507129a3e7cab9d97d10e6790d89512a7db214",
        "fast_foundation_input_resolution": (640, 360),
        "fast_foundation_max_disparity": 192,
        "fast_foundation_valid_iters": 8,
        "fast_foundation_low_memory": True,
        "fast_foundation_alignment": "left",
        "fast_foundation_metric_conversion": "fx_times_baseline_over_disparity",
        "fast_foundation_visibility_mask": "positive_disparity_and_right_correspondence_in_bounds",
    },
)

_dual_config = dataclasses.replace(
    _single_config,
    name="foveavla_dual",
    model=dataclasses.replace(
        _single_config.model,
        image_keys=("base_0_rgb", "left_wrist_0_rgb", "base_1_rgb", "right_wrist_0_rgb"),
        depth_image_keys=("base_0_rgb", "left_wrist_0_rgb", "base_1_rgb", "right_wrist_0_rgb"),
        future_bbox_count=2,
    ),
    data=dataclasses.replace(
        _single_config.data,
        depth_aux_dirs=(_DEPTH_RIGHT,),
        dual_exterior_views=True,
    ),
    policy_metadata=_single_config.policy_metadata
    | {
        "model_name": "FoveaVLA-Dual",
        "droid_exterior_views": ("physical_left", "physical_right"),
        "future_bbox_views": ("physical_left", "physical_right"),
        "metric_depth_views": ("full", "foveal", "full", "foveal"),
    },
)

_CONFIGS_DICT = {_single_config.name: _single_config, _dual_config.name: _dual_config}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
