"""Compute state/action normalization from one finite pass over prepared DROID data."""

from collections.abc import Iterable
import itertools

import numpy as np
import tqdm
import tyro

import foveavla.shared.normalize as normalize
import foveavla.training.config as _config
import foveavla.training.data_loader as _data_loader
import foveavla.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[Iterable[dict], int | None]:
    if batch_size < 1 or (max_frames is not None and max_frames < batch_size):
        raise ValueError("batch_size must be positive and max_frames must be at least batch_size")
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False, repeat=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
        is_batched=True,
    )
    num_batches = max_frames // batch_size if max_frames is not None else None
    return itertools.islice(dataset, num_batches), num_batches


def main(config_name: str, max_frames: int | None = None, batch_size: int = 32):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    data_loader, num_batches = create_rlds_dataloader(data_config, config.model.action_horizon, batch_size, max_frames)
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    frame_count = 0
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in stats:
            stats[key].update(np.asarray(batch[key]))
        frame_count += len(batch["state"])
    if frame_count == 0:
        raise ValueError("No prepared frames matched the RLDS filter; normalization statistics were not written")
    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    output_path = config.assets_dirs / data_config.asset_id
    print(f"Writing stats for {frame_count} frames to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
