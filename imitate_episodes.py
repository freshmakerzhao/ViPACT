import torch
import numpy as np
import os
import glob
import pickle
import argparse
import csv
import json
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT, PUPPET_GRIPPER_JOINT_OPEN, load_config, get_training_config, get_equipment_model, get_sim_task_config
from utils import load_data, build_oracle_static_mask_dict # data functions
from utils import sample_box_pose, sample_box_pose_eval, sample_box_pose_for_excavator, sample_complex_scene_pose_eval, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

import IPython
e = IPython.embed

def main(args):
    set_seed(1)
    
    # 必须指定配置文件路径
    config_path = args.get('config')
    if not config_path:
        raise ValueError("必须通过 --config 参数指定配置文件路径")
    
    # 加载YAML配置
    yaml_config = load_config(config_path)
    training_config = get_training_config(config_path)
    
    # 只从配置文件读取参数
    task_name = yaml_config.get('task', {}).get('name', 'sim_lifting_cube_scripted')
    is_eval = training_config.get('eval', False)
    ckpt_dir = training_config.get('ckpt_dir', './ckpts')
    policy_class = training_config.get('policy_class', 'ACT')
    onscreen_render = training_config.get('onscreen_render', False)
    batch_size_train = training_config.get('batch_size', 32)
    batch_size_val = training_config.get('batch_size', 32)
    num_epochs = training_config.get('num_epochs', 2000)
    clear_videos_before_eval = training_config.get('clear_videos_before_eval', True) # 是否清除对应目录下的mp4视频，默认清除
    mask_ablation = yaml_config.get('eval', {}).get('mask_ablation', 'oracle') # 验证时的mask消融方式，默认不消融（oracle），可选项包括移除（zero）和随机噪声（random）
    eval_output_dir = yaml_config.get('eval', {}).get('output_dir', ckpt_dir) # 评估产物输出目录，默认与ckpt目录一致
    equipment_model = get_equipment_model(config_path)
    seed = training_config.get('seed', 1000)
    use_mask_conditioning = training_config.get('use_mask_conditioning', False)
    image_channels = 4 if use_mask_conditioning else 3
    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        task_config = get_sim_task_config(task_name, config_path)
    else:
        # deploy
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # 双臂14，单臂7，挖掘机4
    if 'bimanual' in equipment_model:
        state_dim = 14
    elif 'excavator_simple' in equipment_model:
        state_dim = 4
    else:
        state_dim = 7

    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': float(training_config.get('lr', 1e-5)),
                         'num_queries': training_config.get('chunk_size', 100),
                         'kl_weight': training_config.get('kl_weight', 10),
                         'hidden_dim': training_config.get('hidden_dim', 512),
                         'dim_feedforward': training_config.get('dim_feedforward', 3200),
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'equipment_model': equipment_model,
                         }
        if use_mask_conditioning:
            policy_config['use_mask_conditioning'] = True
            policy_config['image_channels'] = image_channels
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': float(training_config.get('lr', 1e-5)), 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': training_config.get('lr', 1e-5),
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': seed,
        'temporal_agg': training_config.get('temporal_agg', False),
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'use_mask_conditioning': use_mask_conditioning,
        'image_channels': image_channels,
        'mask_ablation': mask_ablation,
        'eval_output_dir': eval_output_dir,
    }
    if is_eval:
        if not os.path.isdir(eval_output_dir):
            os.makedirs(eval_output_dir, exist_ok=True)
        if clear_videos_before_eval:
            clear_eval_videos(eval_output_dir)
        ckpt_names = [f'policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True, equipment_model=equipment_model)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir,
        num_episodes,
        camera_names,
        batch_size_train,
        batch_size_val,
        use_mask_conditioning=use_mask_conditioning,
    )

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names, use_mask_conditioning=False, static_mask_dict=None):
    if not use_mask_conditioning:
        # Keep the original ACT RGB preprocessing path for strict backward-compatibility.
        curr_images = []
        for cam_name in camera_names:
            curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
            curr_images.append(curr_image)
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        return curr_image

    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w').astype(np.float32) / 255.0
        if static_mask_dict is None or cam_name not in static_mask_dict:
            raise ValueError(f'Missing static mask for camera {cam_name}')
        static_mask = static_mask_dict[cam_name].astype(np.float32)
        curr_image = np.concatenate([curr_image, np.expand_dims(static_mask, axis=0)], axis=0)
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image).float().cuda().unsqueeze(0)
    return curr_image


def apply_mask_ablation(static_mask_dict, mode, seed):
    if mode == 'oracle':
        return static_mask_dict
    if mode == 'zero':
        return {k: np.zeros_like(v, dtype=np.float32) for k, v in static_mask_dict.items()}
    if mode == 'random':
        rng = np.random.default_rng(seed)
        return {
            k: (rng.random(v.shape) > 0.5).astype(np.float32)
            for k, v in static_mask_dict.items()
        }
    raise ValueError(f"Unsupported mask_ablation mode: {mode}. Expected one of ['oracle','zero','random'].")


def clear_eval_videos(ckpt_dir):
    video_paths = glob.glob(os.path.join(ckpt_dir, 'video*.mp4'))
    for video_path in video_paths:
        os.remove(video_path)
    print(f'Cleared {len(video_paths)} existing videos from {ckpt_dir}')

# 回放
def eval_bc(config, ckpt_name, save_episode=True, equipment_model='vx300s_bimanual'):
    set_seed(config['seed'])
    ckpt_dir = config['ckpt_dir']
    eval_output_dir = config.get('eval_output_dir', ckpt_dir)
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    use_mask_conditioning = config.get('use_mask_conditioning', False)
    mask_ablation = config.get('mask_ablation', 'oracle')
    onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name, equipment_model)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    rollout_logs = []
    if use_mask_conditioning:
        print(f'[Eval] mask_ablation={mask_ablation}')
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset
        elif 'sim_lifting_cube' in task_name:
            if 'excavator' in equipment_model:
                BOX_POSE[0] = sample_box_pose_for_excavator()
            elif 'with_complex_scene' in task_name:
                BOX_POSE[0] = sample_complex_scene_pose_eval()
            else:
                BOX_POSE[0] = sample_box_pose_eval()
        else:
            raise NotImplementedError
        init_pose = np.asarray(BOX_POSE[0], dtype=np.float64).copy()
        ts = env.reset()
        static_mask_dict = None
        if use_mask_conditioning:
            static_mask_dict = build_oracle_static_mask_dict(env._physics, camera_names, task_name)
            static_mask_dict = apply_mask_ablation(
                static_mask_dict,
                mode=mask_ablation,
                seed=int(config['seed']) + int(rollout_id),
            )

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(
                    ts,
                    camera_names,
                    use_mask_conditioning=use_mask_conditioning,
                    static_mask_dict=static_mask_dict,
                )

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        success = int(episode_highest_reward == env_max_reward)
        init_x = float(init_pose[0]) if init_pose.size > 0 else np.nan
        init_y = float(init_pose[1]) if init_pose.size > 1 else np.nan
        init_z = float(init_pose[2]) if init_pose.size > 2 else np.nan
        rollout_logs.append({
            'rollout_id': int(rollout_id),
            'task_name': task_name,
            'equipment_model': equipment_model,
            'init_x': init_x,
            'init_y': init_y,
            'init_z': init_z,
            'init_pose': init_pose.tolist(),
            'episode_return': float(episode_return),
            'episode_highest_reward': float(episode_highest_reward),
            'env_max_reward': float(env_max_reward),
            'success': success,
        })
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            success_tag = 'Success' if success == 1 else 'Failure'
            video_name = f'video{rollout_id}_r{int(episode_highest_reward)}_ret{episode_return:.2f}_{success_tag}.mp4'
            save_videos(image_list, DT, video_path=os.path.join(eval_output_dir, video_name))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(eval_output_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))
        f.write('\n\n')
        f.write('run_config:\n')
        f.write(json.dumps(config, ensure_ascii=False, indent=2))

    rollout_log_file_name = 'result_' + ckpt_name.split('.')[0] + '_rollout_log.csv'
    rollout_log_path = os.path.join(eval_output_dir, rollout_log_file_name)
    with open(rollout_log_path, 'w', newline='') as csvfile:
        fieldnames = [
            'rollout_id', 'task_name', 'equipment_model',
            'init_x', 'init_y', 'init_z', 'init_pose',
            'episode_return', 'episode_highest_reward', 'env_max_reward', 'success'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rollout_logs:
            writer.writerow(row)
    print(f'Rollout log saved to: {rollout_log_path}')

    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    # ==== debug ====
    # 检查数据维度
    # print(image_data.shape, qpos_data.shape, action_data.shape, is_pad.shape)
    # ==== debug ====
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ACT训练和评估脚本')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径 (必须指定，如 configs/fairino5_single/02_train.yaml 或 configs/fairino5_single/03_eval.yaml)')
    
    args = parser.parse_args()
    
    main(vars(args))
