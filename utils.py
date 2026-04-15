import numpy as np
import torch
import os
import h5py
import mujoco
import random
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

def get_target_geom_names(task_name):
    if 'sim_transfer_cube' in task_name:
        return ['red_box']
    if 'sim_lifting_cube' in task_name:
        return ['red_box']
    if 'sim_insertion' in task_name:
        return ['red_peg']
    raise NotImplementedError(f'No target geom mapping for task_name={task_name}')


def build_oracle_static_mask_dict(physics, camera_names, task_name, height=480, width=640):
    """Build per-camera static masks from MuJoCo segmentation (oracle)."""
    target_geom_ids = []
    for geom_name in get_target_geom_names(task_name):
        try:
            target_geom_ids.append(physics.model.name2id(geom_name, 'geom'))
        except (KeyError, ValueError):
            continue
    if len(target_geom_ids) == 0:
        raise ValueError(f'No valid target geom id found for task_name={task_name}')
    target_geom_ids = np.array(target_geom_ids, dtype=np.int32)
    geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)

    static_mask_dict = {}
    for cam_name in camera_names:
        segmentation = physics.render(
            height=height, width=width, camera_id=cam_name, segmentation=True
        )
        # dm_control segmentation format: [..., 0]=objid, [..., 1]=objtype
        objid = segmentation[..., 0]
        objtype = segmentation[..., 1]
        mask = np.isin(objid, target_geom_ids) & (objtype == geom_type)
        mask = mask.astype(np.float32)
        if mask.sum() < 1:
            raise RuntimeError(
                f'Oracle mask is empty for camera={cam_name}, task={task_name}. '
                'Please verify target visibility or camera selection.'
            )
        static_mask_dict[cam_name] = mask

    return static_mask_dict


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, use_mask_conditioning=False):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.use_mask_conditioning = use_mask_conditioning # 是否使用掩码条件，掩码在整个episode中保持静态（即使动作和qpos在变化），以提供对物体位置的持续感知
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            static_mask_dict = dict()
            has_oracle_static_masks = (
                'observations' in root and
                'static_masks' in root['observations']
            )
            if self.use_mask_conditioning and not has_oracle_static_masks:
                raise KeyError(
                    f'Dataset {dataset_path} is missing /observations/static_masks. '
                    'Please regenerate dataset with record_sim_episodes.py to use oracle masks.'
                )
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
                if self.use_mask_conditioning:
                    if cam_name not in root['/observations/static_masks']:
                        raise KeyError(
                            f'Dataset {dataset_path} has no static mask for camera {cam_name}'
                        )
                    static_mask_dict[cam_name] = root[f'/observations/static_masks/{cam_name}'][()]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)
        # 生成静态掩码字典，并为每个摄像头构建掩码列表
        if self.use_mask_conditioning:
            all_cam_masks = []
            for cam_name in self.camera_names:
                all_cam_masks.append(static_mask_dict[cam_name])
            all_cam_masks = np.stack(all_cam_masks, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data.float() / 255.0
        # 将掩码作为额外的通道连接到图像数据中
        if self.use_mask_conditioning:
            mask_data = torch.from_numpy(all_cam_masks).float().unsqueeze(1)
            image_data = torch.cat([image_data, mask_data], dim=1)
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, use_mask_conditioning=False):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(
        train_indices, dataset_dir, camera_names, norm_stats, use_mask_conditioning=use_mask_conditioning
    )
    val_dataset = EpisodicDataset(
        val_indices, dataset_dir, camera_names, norm_stats, use_mask_conditioning=use_mask_conditioning
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_box_pose_eval():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

# 生成复杂场景的目标和干扰物位姿，确保它们之间有足够的距离以避免重叠
def _sample_cube_pose_from_range(x_range, y_range, z_range):
    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

# 在复杂场景中采样目标和干扰物位姿，确保它们之间有足够的距离以避免重叠
def _sample_non_overlapping_cube_pose(existing_xyz_list, x_range, y_range, z_range, min_dist=0.07, max_trials=2000):
    for _ in range(max_trials):
        pose = _sample_cube_pose_from_range(x_range, y_range, z_range)
        xyz = pose[:3]
        if all(np.linalg.norm(xyz - prev_xyz) >= min_dist for prev_xyz in existing_xyz_list):
            return pose
    raise RuntimeError('Failed to sample non-overlapping cube pose in complex scene')

# 生成复杂场景的目标和干扰物位姿，确保它们之间有足够的距离以避免重叠
def sample_complex_scene_pose():
    """Sample target + 3 distractor cube poses for complex lifting scene.

    Return shape: (28,) = 4 boxes * (xyz + quat).
    Order is fixed to keep scripted policy and replay logic aligned:
    [red_box, distractor_box_1, distractor_box_2, distractor_box_3].
    """
    target_pose = sample_box_pose()

    distractor_x_range = [-0.12, 0.32]
    distractor_y_range = [0.38, 0.70]
    z_range = [0.05, 0.05]

    xyz_list = [target_pose[:3]]
    distractors = []
    for _ in range(3):
        d_pose = _sample_non_overlapping_cube_pose(
            xyz_list,
            distractor_x_range,
            distractor_y_range,
            z_range,
            min_dist=0.07,
        )
        distractors.append(d_pose)
        xyz_list.append(d_pose[:3])

    return np.concatenate([target_pose] + distractors)

# 评估时的复杂场景采样，保持与训练分布一致
def sample_complex_scene_pose_eval():
    # Keep eval distribution aligned with train for current MVP.
    return sample_complex_scene_pose()


def sample_box_pose_eval_ring():
    outer_x_range = [-0.1, 0.3]
    outer_y_range = [0.3, 0.7]
    inner_x_range = [0.0, 0.2]
    inner_y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    outer_ranges = np.vstack([outer_x_range, outer_y_range, z_range])

    for _ in range(1000):
        cube_position = np.random.uniform(outer_ranges[:, 0], outer_ranges[:, 1])
        x, y = cube_position[0], cube_position[1]
        in_inner = (inner_x_range[0] <= x <= inner_x_range[1]) and (inner_y_range[0] <= y <= inner_y_range[1])
        if not in_inner:
            cube_quat = np.array([1, 0, 0, 0])
            return np.concatenate([cube_position, cube_quat])

    raise RuntimeError('Failed to sample box pose in eval ring after 1000 attempts')

def sample_box_pose_for_excavator():
    x_range = [3.4, 4.4]
    y_range = [-1, 1]
    z_range = [0.25, 0.25]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    np.random.seed(seed)
