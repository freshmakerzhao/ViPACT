import pathlib
import numpy as np
import yaml
import os

def load_config(config_path):
    """从YAML文件加载配置"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            print(f"[DEBUG] 成功加载配置文件: {config_path}")
            print(f"[DEBUG] 配置内容: {config}")
            return config
    else:
        print(f"[ERROR] 配置文件不存在: {config_path}")
        return {}

def get_task_config(config_path):
    """从YAML配置获取任务配置"""
    config = load_config(config_path)
    if config and 'task' in config:
        task_config = config['task']
        return {
            'dataset_dir': task_config.get('dataset_dir', ''),
            'num_episodes': task_config.get('num_episodes', 50),
            'episode_len': task_config.get('episode_len', 400),
            'camera_names': task_config.get('camera_names', ['top'])
        }
    return None

def get_equipment_model(config_path):
    """从YAML配置获取设备型号"""
    config = load_config(config_path)
    if config and 'equipment' in config:
        return config['equipment'].get('model', 'vx300s_bimanual')
    return 'vx300s_bimanual'

def get_training_config(config_path):
    """从YAML配置获取训练配置"""
    config = load_config(config_path)
    if config:
        training = config.get('training', {})
        act = config.get('act', {})
        output = config.get('output', {})
        eval_config = config.get('eval', {})
        vipact = config.get('vipact', {})

        use_mask_conditioning = bool(vipact.get('use_mask_conditioning', False))
        image_channels = 4 if use_mask_conditioning else 3
        
        return {
            'policy_class': training.get('policy_class', 'ACT'),
            'batch_size': training.get('batch_size', 32),
            'num_epochs': training.get('num_epochs', 2000),
            'val_every': training.get('val_every', 1),
            'lr': training.get('lr', 1e-5),
            'seed': training.get('seed', 1000),
            'temporal_agg': training.get('temporal_agg', False),
            'kl_weight': act.get('kl_weight', 10),
            'chunk_size': act.get('chunk_size', 100),
            'hidden_dim': act.get('hidden_dim', 512),
            'dim_feedforward': act.get('dim_feedforward', 3200),
            'ckpt_dir': output.get('ckpt_dir', './ckpts'),
            'eval': eval_config.get('enabled', False),
            'clear_videos_before_eval': eval_config.get('clear_videos_before_eval', True),
            'use_mask_conditioning': use_mask_conditioning,
            'image_channels': image_channels,
        }
    return {}

### Task parameters
DATA_DIR = '/home/zhaoshuai/workspace_act/PACT/data_sim_episodes/26032502_fairino5_single_act_origin' # absolute path
SIM_TASK_CONFIGS = {
    # vx300s_bimanual搬方块任务
    'sim_transfer_cube_scripted':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_transfer_cube_human':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_human',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_scripted': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_human': {
        'dataset_dir': DATA_DIR + '/sim_insertion_human',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['top']
    },

    # 单臂搬方块任务
    # 'camera_names': ['top', 'angle', 'right_pillar']
    'sim_lifting_cube_scripted':{
        'dataset_dir': DATA_DIR + '/sim_lifting_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },
    'sim_lifting_cube_with_complex_scene_scripted':{
        'dataset_dir': DATA_DIR + '/sim_lifting_cube_with_complex_scene_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },
}

def get_sim_task_config(task_name, config_path):
    """获取任务配置，优先从YAML文件加载"""
    yaml_config = get_task_config(config_path)
    if yaml_config:
        return yaml_config
    # 如果YAML配置不存在，使用默认配置
    return SIM_TASK_CONFIGS.get(task_name, SIM_TASK_CONFIGS['sim_lifting_cube_scripted'])

### Simulation envs fixed constants
DT = 0.02

# ========================= vx300s ==================================
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239] # 双臂初始位置，前6+2维是左臂，后6+2维是右臂
START_SINGLE_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239] # vx300s 单臂初始位置
# ========================= vx300s ==================================

# ========================= fairino single ==================================
START_FAIRINO_POSE_origin = [0, 0, 0, 0, 0, 0, 0.057, -0.057]
START_FAIRINO_POSE = [0, -1.5708, 1.5708, -1.5708, -1.5708, 0, 0.057, -0.057]
# ========================= fairino single==================================

# ========================= 挖掘机 ==================================
EXCAVATOR_MAIN_JOINTS = ('j1_swing', 'j2_boom', 'j3_stick', 'j4_bucket')
EXCAVATOR_START_POSE = np.array([0.0, -0.25, -0.5, -0.5])
# ========================= 挖掘机 ==================================

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path

# 夹爪开合距离，master是操作者，puppet是搖操的或仿真中的夹爪
# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# 夹爪关节开合角度，master是操作者，puppet是搖操的或仿真中的夹爪
# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

# 归一化，根据夹爪开合距离，将实际夹爪位置映射到[0, 1]区间
MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
# 反归一化，根据夹爪开合距离，将[0, 1]区间的值映射回实际夹爪位置
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
# 将master夹爪位置映射到puppet夹爪位置
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
