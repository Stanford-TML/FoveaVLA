"""DROID RLDS decoding, frame filtering, and action chunking."""

from collections.abc import Sequence
import dataclasses
import json
import logging
from pathlib import Path

import tqdm

import foveavla.shared.download as download


@dataclasses.dataclass
class RLDSDataset:
    name: str
    version: str
    weight: float
    filter_dict_path: str | None = None


class DroidRldsDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        datasets: Sequence[RLDSDataset],
        *,  # Force keyword-only arguments
        shuffle: bool = True,
        repeat: bool = True,
        action_chunk_size: int = 16,
        dual_exterior_views: bool = False,
        # Reduce this if you are running out of memory, but careful -- below ~100k shuffling is not sufficiently random.
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
    ):
        # Import tensorflow here to not make it mandatory in case RLDS data loader is not used.
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds

        # Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch / JAX)
        tf.config.set_visible_devices([], "GPU")

        # Ensure dataset weights sum to 1.0
        assert sum(dataset.weight for dataset in datasets) == 1.0, "Dataset weights must sum to 1.0"
        def decode_images(frame):
            exterior_image_keys = ("left_image", "right_image") if dual_exterior_views else ("image",)
            for image_key in exterior_image_keys:
                frame["observation"][image_key] = tf.io.decode_image(
                    frame["observation"][image_key], expand_animations=False, dtype=tf.uint8
                )
            return frame

        def prepare_single_dataset(dataset_cfg: RLDSDataset):
            ds_name, version = dataset_cfg.name, dataset_cfg.version
            builder = tfds.builder(ds_name, data_dir=data_dir, version=version)
            dataset = dl.DLataset.from_rlds(
                builder, split="train", shuffle=shuffle, num_parallel_reads=num_parallel_reads
            )

            # Filter out any unsuccessful trajectories -- we use the file name to check this
            dataset = dataset.filter(
                lambda traj: tf.strings.regex_full_match(
                    traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
                )
            )

            # Repeat dataset so we never run out of data.
            if repeat:
                dataset = dataset.repeat()

            # Load the filter dictionary if provided.
            # The filter dictionary is a JSON file that maps episode keys to ranges of frames to sample
            # (e.g.,
            # {
            #     "<episode key>": [[0, 100], [200, 300]]
            # }
            # means keep frames 0-99 and 200-299).

            filter_dict_path = dataset_cfg.filter_dict_path
            if filter_dict_path is not None:
                cached_filter_dict_path = download.maybe_download(filter_dict_path)
                with Path(cached_filter_dict_path).open("r") as f:
                    filter_dict = json.load(f)
                logging.info(f"Using filter dictionary with {len(filter_dict)} episodes")

                keys_tensor = []
                values_tensor = []

                for episode_key, ranges in tqdm.tqdm(filter_dict.items(), desc="Creating idle filter hash table..."):
                    for start, end in ranges:
                        for t in range(start, end):
                            frame_key = f"{episode_key}--{t}"
                            keys_tensor.append(frame_key)
                            values_tensor.append(True)
                self.filter_table = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor), default_value=False
                )
                logging.info("Filter hash table initialized")
            else:
                self.filter_table = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer([""], [True]), default_value=True
                )

            def restructure(traj):
                """Reformat observation and action keys, sample language instruction."""
                traj_len = tf.shape(traj["action"])[0]
                # Important: we use joint *position* action space -- easier to simulate!
                actions = tf.concat(
                    (
                        traj["action_dict"]["joint_position"],
                        traj["action_dict"]["gripper_position"],
                    ),
                    axis=-1,
                )
                # Randomly sample one of the three language instructions
                instruction = tf.random.shuffle(
                    [traj["language_instruction"], traj["language_instruction_2"], traj["language_instruction_3"]]
                )[0]

                indices = tf.as_string(tf.range(traj_len))

                # Data filtering:
                # Compute a uniquely-identifying step ID by concatenating the recording folderpath, file path,
                # and each step's time step index. This will index into the filter hash table, and if it returns true,
                # then the frame passes the filter.
                step_id = (
                    traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
                    + "--"
                    + traj["traj_metadata"]["episode_metadata"]["file_path"]
                    + "--"
                    + indices
                )
                passes_filter = self.filter_table.lookup(step_id)

                observation = {
                    "joint_position": traj["observation"]["joint_position"],
                    "gripper_position": traj["observation"]["gripper_position"],
                }
                if dual_exterior_views:
                    observation["left_image"] = traj["observation"]["exterior_image_1_left"]
                    observation["right_image"] = traj["observation"]["exterior_image_2_left"]
                else:
                    observation["image"] = traj["observation"]["exterior_image_1_left"]

                result = {
                    "actions": actions,
                    "observation": observation,
                    "prompt": instruction,
                    "step_id": step_id,
                    "passes_filter": passes_filter,
                }
                result["episode_path"] = traj["traj_metadata"]["episode_metadata"]["file_path"]
                result["frame_idx"] = tf.range(traj_len, dtype=tf.int32)
                if not dual_exterior_views:
                    result["exterior_view"] = tf.fill([traj_len], "exterior_image_1_left")
                return result

            dataset = dataset.traj_map(restructure, num_parallel_calls)

            def chunk_actions(traj):
                """Splits episode into action chunks."""
                traj_len = tf.shape(traj["actions"])[0]

                # For each step in the trajectory, construct indices for the next n actions
                action_chunk_indices = tf.broadcast_to(
                    tf.range(action_chunk_size)[None],
                    [traj_len, action_chunk_size],
                ) + tf.broadcast_to(
                    tf.range(traj_len)[:, None],
                    [traj_len, action_chunk_size],
                )

                # Cap to length of the sequence --> final chunks will repeat the last action
                # This makes sense, since we are using absolute joint + gripper position actions
                action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)

                # Gather the actions for each chunk
                traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
                return traj

            dataset = dataset.traj_map(chunk_actions, num_parallel_calls)

            # Flatten: map from trajectory dataset to dataset of individual action chunks
            dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

            # Filter data that doesn't pass the filter
            def filter_from_dict(frame):
                return frame["passes_filter"]

            dataset = dataset.filter(filter_from_dict)

            # Remove "passes_filter" key from output
            def remove_passes_filter(frame):
                frame.pop("passes_filter")
                return frame

            return dataset.map(remove_passes_filter)

        logging.info(f"Preparing {len(datasets)} datasets...")
        logging.info("-" * 50)
        for dataset in datasets:
            logging.info(f"    {dataset.name}:{dataset.version} with weight {dataset.weight:.2f}")
        logging.info("-" * 50)
        all_datasets = [prepare_single_dataset(dataset) for dataset in datasets]
        weights = [dataset.weight for dataset in datasets]

        final_dataset = dl.DLataset.sample_from_datasets(all_datasets, weights=weights)
        if shuffle:
            final_dataset = final_dataset.shuffle(shuffle_buffer_size)
        # Keep compressed JPEGs through filtering and the large shuffle buffer; decode only selected frames.
        final_dataset = final_dataset.frame_map(decode_images, num_parallel_calls)
        final_dataset = final_dataset.batch(batch_size)
        # Note =>> Seems to reduce memory usage without affecting speed?
        final_dataset = final_dataset.with_ram_budget(1)

        self.dataset = final_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()
