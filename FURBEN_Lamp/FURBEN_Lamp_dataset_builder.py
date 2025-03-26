from typing import Iterator, Tuple, Any

import os
import h5py
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
from LIBERO_Spatial.conversion_utils import MultiThreadedDatasetBuilder
import pickle


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    # the line below needs to be *inside* generate_examples so that each worker creates it's own model
    # creating one shared model outside this function would cause a deadlock

    def _parse_example(episode_path, demo_id):
        # load raw data
        with open(episode_path, "rb") as f:
            data = pickle.load(f)
            print("episode_path", episode_path)
            actions = np.stack(data["actions"], axis=0)

            concat_order = ["ee_pos", "ee_quat", "ee_pos_vel", "ee_ori_vel", "joint_positions", "joint_velocities", "joint_torques", "gripper_width"]
            states = np.stack([np.concatenate([data["observations"][i]["robot_state"][k].reshape(-1) for k in concat_order], axis=-1) for i in range(len(data["observations"]))], axis=0)
            
            wrist_images = np.stack([data["observations"][i]["color_image1"] for i in range(len(data["observations"]))], axis=0)
            front_images = np.stack([data["observations"][i]["color_image2"] for i in range(len(data["observations"]))], axis=0)
            rear_images = np.stack([data["observations"][i]["color_image3"] for i in range(len(data["observations"]))], axis=0)
            wrist_segs = np.expand_dims(np.stack([data["observations"][i]["seg_image1"] for i in range(len(data["observations"]))], axis=0), axis=-1)
            front_segs = np.expand_dims(np.stack([data["observations"][i]["seg_image2"] for i in range(len(data["observations"]))], axis=0), axis=-1)
            rear_segs = np.expand_dims(np.stack([data["observations"][i]["seg_image3"] for i in range(len(data["observations"]))], axis=0), axis=-1)
            rewards = np.stack(data["rewards"])
            skils = np.stack(data["skills"])
            furniture = data["furniture"]
            initial_randomness = "low"

        # compute language instruction
        command = "Assembly a lamp with a lamp base, a lamp bulb, and a lamp hood."
        # assemble episode --> here we're assuming demos so we set reward to 1 at the end
        episode = []
        for i in range(actions.shape[0]):
            episode.append({
                'observation': {
                    'wrist_image': wrist_images[i],
                    'front_image': front_images[i],
                    'rear_image': rear_images[i],
                    'wrist_seg': wrist_segs[i],
                    'front_seg': front_segs[i],
                    'rear_seg': rear_segs[i],
                    'state': np.asarray(states[i], np.float32),
                },
                'action': np.asarray(actions[i], dtype=np.float32),
                'discount': 1.0,
                'reward': rewards[i],
                'is_first': i == 0,
                'is_last': i == (actions.shape[0] - 1),
                'is_terminal': i == (actions.shape[0] - 1),
                'language_instruction': command,
                'skill_completion': np.asarray(skils[i], dtype=np.float32),
            })

        # create output data sample
        sample = {
            'steps': episode,
            'episode_metadata': {
                'furniture': furniture,
                'file_path': episode_path,
                'episode_id': "0",
                'initial_randomness': initial_randomness,
            }
        }

        # if you want to skip an example for whatever reason, simply return None
        return episode_path + f"_{demo_id}", sample

    # for smallish datasets, use single-thread parsing
    for sample in paths:
        n_demos = 1 # furniturebench data 1 demo for each file
        idx = 0
        cnt = 0
        while cnt < n_demos:
            ret = _parse_example(sample, idx)
            if ret is not None:
                cnt += 1
            idx += 1
            yield ret


class FURBENLamp(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    N_WORKERS = 40             # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 80   # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'wrist_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera RGB observation.',
                        ),
                        'front_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Front camera RGB observation.',
                        ),
                        'rear_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Rear camera RGB observation.',
                        ),
                        'wrist_seg': tfds.features.Tensor(
                            shape=(256, 256, 1),
                            dtype=np.int32,
                            doc='Wrist camera segmentation.',
                        ),
                        'front_seg': tfds.features.Tensor(
                            shape=(256, 256, 1),
                            dtype=np.int32,
                            doc='Front camera segmentation.',
                        ),
                        'rear_seg': tfds.features.Tensor(
                            shape=(256, 256, 1),
                            dtype=np.int32,
                            doc='Rear camera segmentation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(37,),
                            dtype=np.float32,
                            doc='Robot state, consists of [3x eef position, 4x eef quaternion, 3x eef linear velocity, 3x eef angular velocity, 7x joint position, 7x joint velocity, 7x joint torque, 1x gripper width].',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(8,),
                        dtype=np.float32,
                        doc='Robot EEF action (3D trans + 4D rot + 1D open/close).',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'skill_completion': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='+1 skill completion reward; otherwise, 0.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'furniture': tfds.features.Text(
                        doc='Furniture model name.'
                    ),
                    'initial_randomness': tfds.features.Text(
                        doc='Randomness in furniture initial configuration.[low, med, high]'
                    ),
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'episode_id': tfds.features.Text(
                        doc='Episode ID.'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        return {
            "train": glob.glob("/data2/lzixuan/furniture-bench/scripted_sim_demo/lamp/*/*.pkl"),
        }
