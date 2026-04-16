import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py

from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS, load_config, get_equipment_model, get_sim_task_config
from ee_sim_env import make_ee_sim_env
from sim_env import make_sim_env, BOX_POSE
from scripted_policy import PickAndTransferPolicy, InsertionPolicy, LiftingAndMovingPolicy, ExcavatorMocapLiftingPolicy
from utils import build_oracle_static_mask_dict, resolve_target_geom_name, get_complex_scene_target_geom_names, sample_complex_scene_pose

import IPython
e = IPython.embed

TASK_SIM_TRANSFER = 'sim_transfer_cube_scripted'
TASK_SIM_INSERTION = 'sim_insertion_scripted'
TASK_SIM_LIFTING = 'sim_lifting_cube_scripted'
TASK_SIM_LIFTING_COMPLEX = 'sim_lifting_cube_with_complex_scene_scripted'


def main(args):
    """
    生成仿真演示数据（轨迹录制）。
    1) 在 ee_sim_env 中执行末端执行器(EE)空间策略，得到关节轨迹。
    2) 用夹爪控制量替换夹爪关节位置。
    3) 在 sim_env 中重放关节轨迹并记录观测。
    4) 保存为一条 episode 数据并继续下一条。
    """

    # 必须指定配置文件路径
    config_path = args.get('config')
    if not config_path:
        raise ValueError("必须通过 --config 参数指定配置文件路径")
    
    # 加载YAML配置
    yaml_config = load_config(config_path)
    
    # 只从配置文件读取参数
    task_name = yaml_config.get('task', {}).get('name', 'sim_lifting_cube_scripted')
    dataset_dir = yaml_config.get('task', {}).get('dataset_dir', './data_sim_episodes')
    num_episodes = yaml_config.get('task', {}).get('num_episodes', 50)
    onscreen_render = yaml_config.get('render', {}).get('onscreen_render', False)
    equipment_model = get_equipment_model(config_path)
    inject_noise = False
    render_cam_name = 'angle'
    arm_nums = 2 # 默认双臂任务，特殊型号或任务时修改
    is_excavator = equipment_model == 'excavator_simple'

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    # 从任务配置中读取 episode 长度与相机名称
    task_config = get_sim_task_config(task_name, config_path)
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    # ViPACT 固定策略：仅 cockpit 生成 oracle mask；没有 cockpit 时不生成任何 oracle mask。
    mask_camera_names = ['cockpit'] if 'cockpit' in camera_names else []
    print(f'[Record] mask_camera_names={mask_camera_names}')
    is_complex_lifting = task_name == TASK_SIM_LIFTING_COMPLEX
    # target_conditioning 是复杂搬运任务的特殊配置项，默认关闭，开启后会根据 target_id 切换不同目标物体进行训练
    target_conditioning_cfg = yaml_config.get('target_conditioning', {})
    target_conditioning_enabled = bool(target_conditioning_cfg.get('enabled', False))
    target_sampling_mode = target_conditioning_cfg.get('sampling', 'cycle')
    counterfactual_same_layout = bool(target_conditioning_cfg.get('counterfactual_same_layout', False))
    if target_sampling_mode not in ('cycle', 'random'):
        # cycle：按 0,1,2,3,0,1... 轮询，分布均匀、可复现。
        # random：每条随机选一个目标。
        raise ValueError(f'Unsupported target_conditioning.sampling={target_sampling_mode}, expected cycle/random')
    if target_conditioning_enabled and not is_complex_lifting:
        print('[WARN] target_conditioning.enabled=True but task is not complex lifting. Fallback to single target.')
        target_conditioning_enabled = False
    if counterfactual_same_layout and not is_complex_lifting:
        print('[WARN] target_conditioning.counterfactual_same_layout=True but task is not complex lifting. Disabled.')
        counterfactual_same_layout = False
    if counterfactual_same_layout and not target_conditioning_enabled:
        print('[WARN] counterfactual_same_layout=True requires target_conditioning.enabled=True. Disabled.')
        counterfactual_same_layout = False
    # 获取target_id与target_geom_name的映射关系，复杂搬运任务有多个目标物体可选，其他任务默认单目标
    complex_target_geom_names = get_complex_scene_target_geom_names() if is_complex_lifting else []
    if counterfactual_same_layout and len(complex_target_geom_names) == 0:
        raise ValueError('counterfactual_same_layout=True requires non-empty complex target geom list')
    if task_name == TASK_SIM_TRANSFER:
        policy_cls = PickAndTransferPolicy
    elif task_name == TASK_SIM_INSERTION:
        policy_cls = InsertionPolicy
    elif task_name in (TASK_SIM_LIFTING, TASK_SIM_LIFTING_COMPLEX):
        if equipment_model == 'excavator_simple':
            policy_cls = ExcavatorMocapLiftingPolicy
        else:
            policy_cls = LiftingAndMovingPolicy
        arm_nums = 1
    else:
        raise NotImplementedError

    if is_excavator:
        state_dim = 4
    else:
        state_dim = 14 if arm_nums == 2 else 7

    layout_pose_cache = []
    if counterfactual_same_layout:
        targets_per_layout = len(complex_target_geom_names)
        num_layouts = (num_episodes + targets_per_layout - 1) // targets_per_layout
        for _ in range(num_layouts):
            layout_pose_cache.append(sample_complex_scene_pose())
        print(
            f'[Record] counterfactual_same_layout enabled: '
            f'num_layouts={num_layouts}, targets_per_layout={targets_per_layout}, num_episodes={num_episodes}'
        )

    success = []
    for episode_idx in range(num_episodes):
        fixed_layout_pose = None
        layout_id = -1
        # 复杂搬运任务：可选“同一布局下遍历目标”反事实数据模式
        if counterfactual_same_layout:
            targets_per_layout = len(complex_target_geom_names)
            layout_id = int(episode_idx // targets_per_layout)
            target_id = int(episode_idx % targets_per_layout)
            fixed_layout_pose = np.asarray(layout_pose_cache[layout_id], dtype=np.float64).copy()
        # 复杂搬运任务根据 target_id 切换目标物体，生成多样化数据；其他任务默认单目标不变
        elif target_conditioning_enabled:
            if target_sampling_mode == 'random':
                target_id = int(np.random.randint(0, len(complex_target_geom_names)))
            else:
                target_id = int(episode_idx % len(complex_target_geom_names))
        else:
            target_id = 0
        target_geom_name = resolve_target_geom_name(task_name, target_id)

        print(f'{episode_idx=}')
        if layout_id >= 0:
            print(f'[Record] layout_id={layout_id}, target_id={target_id}, target_geom_name={target_geom_name}')
        else:
            print(f'[Record] target_id={target_id}, target_geom_name={target_geom_name}')
        print('Rollout out EE space scripted policy')
        # 第一阶段：在 EE 空间执行脚本策略，得到关节轨迹
        # setup the environment
        env = make_ee_sim_env(task_name, equipment_model=equipment_model)
        if hasattr(env, 'task'):
            env.task.current_target_id = target_id
            env.task.target_geom_name = target_geom_name
            if fixed_layout_pose is not None:
                env.task.fixed_scene_pose = fixed_layout_pose
        ts = env.reset()
        episode = [ts]
        if policy_cls is LiftingAndMovingPolicy:
            policy = policy_cls(inject_noise, target_id=target_id)
        else:
            policy = policy_cls(inject_noise)
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for step in range(episode_len):
            # 这里action拿到的是当前ts下计算得到的xyz、quat、gripper,一维，size=8
            action = policy(ts) # 这里调用__call__方法
            # EE 环境内部会将 EE 动作转为关节控制并推进仿真
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.002)
        plt.close()
        # ts.reward 是当前时间步的奖励，计算 episode return 时跳过第一个时间步
        episode_return = np.sum([ts.reward for ts in episode[1:]]) # 求reward和
        episode_max_reward = np.max([ts.reward for ts in episode[1:]]) # 求reward最大值
        if episode_max_reward == env.task.max_reward:
            # timestep中有一个达到了最大奖励，说明任务成功
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")

        # 从 EE 环境中抽取关节轨迹(qpos)
        joint_traj = [ts.observation['qpos'] for ts in episode]
        # 夹爪控制量替换夹爪关节位置（避免 EE 环境内部的夹爪状态不一致）
        if not is_excavator:
            gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
            if arm_nums == 2:
                for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
                    left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
                    right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
                    joint[6] = left_ctrl
                    joint[6+7] = right_ctrl
            elif arm_nums == 1:
                for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
                    right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
                    joint[6] = right_ctrl
            else:
                raise NotImplementedError

        # 保存初始环境状态（如物体位姿），确保重放时一致
        subtask_info = episode[0].observation['env_state'].copy() # box pose at step 0

        # clear unused variables
        del env
        del episode
        del policy

        # 第二阶段：在 sim_env 中重放关节轨迹并录制观测
        # setup the environment
        print('Replaying joint commands')
        env = make_sim_env(task_name, equipment_model=equipment_model)
        if hasattr(env, 'task'):
            env.task.current_target_id = target_id
            env.task.target_geom_name = target_geom_name
        # 将物体初始位姿同步到 sim_env
        BOX_POSE[0] = subtask_info # make sure the sim_env has the same object configurations as ee_sim_env
        ts = env.reset()
        # 获取静态掩码，用于后续训练中去除不必要的背景干扰，提升模型专注于目标物体和机械臂的学习效果
        static_mask_dict = build_oracle_static_mask_dict(
            env._physics,
            mask_camera_names,
            task_name,
            target_geom_name=target_geom_name,
            allow_empty_masks=True,
        )

        episode_replay = [ts]
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for t in range(len(joint_traj)): # note: this will increase episode length by 1
            # 直接把关节轨迹作为动作输入 sim_env
            action = joint_traj[t]
            ts = env.step(action)
            episode_replay.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.02)

        episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
        if episode_max_reward == env.task.max_reward:
            success.append(1)
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            success.append(0)
            print(f"{episode_idx=} Failed")

        plt.close()

        """
        For each timestep:
        observations
        - images
            - each_cam_name     (480, 640, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'

        action                  (14,)         'float64'
        """

        # 组织数据：观测(qpos/qvel/图像)与动作(action)
        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []
        for cam_name in mask_camera_names:
            data_dict[f'/observations/static_masks/{cam_name}'] = static_mask_dict[cam_name].astype(np.uint8)

        # 因为重放会多出 1 个动作与 1 个时间步，这里截断保持一致
        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]

        # len(joint_traj) i.e. actions: max_timesteps
        # len(episode_replay) i.e. time steps: max_timesteps + 1
        max_timesteps = len(joint_traj)
        # 按时间步对齐 action 与 observation
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        # 写入 HDF5 数据文件
        t0 = time.time()
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            root.attrs['target_id'] = int(target_id)
            root.attrs['target_geom_name'] = str(target_geom_name)
            root.attrs['target_conditioning_enabled'] = bool(target_conditioning_enabled)
            root.attrs['layout_id'] = int(layout_id)
            root.attrs['counterfactual_same_layout'] = bool(counterfactual_same_layout)
            obs = root.create_group('observations')
            image = obs.create_group('images')
            static_masks = obs.create_group('static_masks')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            for cam_name in mask_camera_names:
                _ = static_masks.create_dataset(cam_name, (480, 640), dtype='uint8')
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            qpos = obs.create_dataset('qpos', (max_timesteps, state_dim))
            qvel = obs.create_dataset('qvel', (max_timesteps, state_dim))
            action = root.create_dataset('action', (max_timesteps, state_dim))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')

    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='轨迹录制脚本，只能通过YAML配置文件加载配置')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径 (必须指定，如 configs/fairino5_single/01_record.yaml)')
    
    args = parser.parse_args()
    
    main(vars(args))
