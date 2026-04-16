from curses.ascii import ctrl
import numpy as np
import collections
import os

from constants import DT, START_FAIRINO_POSE, XML_DIR, START_ARM_POSE, START_SINGLE_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_CLOSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

from constants import EXCAVATOR_MAIN_JOINTS, EXCAVATOR_START_POSE

from utils import sample_box_pose, sample_complex_scene_pose, sample_insertion_pose, sample_box_pose_for_excavator
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

import IPython
e = IPython.embed

def make_ee_sim_env(task_name, equipment_model: str = 'vx300s_bimanual'):
    """
    Environment for simulated robot bi-manual manipulation, with end-effector control.
    Action space:      [left_arm_pose (7),             # position and quaternion for end effector
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_pose (7),            # position and quaternion for end effector
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    # 根据不同任务加载不同的 XML 和 Task
    if 'sim_transfer_cube' in task_name:
        xml_path = os.path.join(XML_DIR, equipment_model, 'bimanual_viperx_ee_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path) # 加载物理引擎，其能够读取关节位置、推进一个step、渲染图像
        task = TransferCubeEETask(random=False) # 任务对象，初始化基本任务设置、奖励函数、观测数据提取
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
        # 这里DT表示控制时间步长，即每隔DT秒执行一次动作，在constants.py中定义为0.02秒
    elif 'sim_insertion' in task_name:
        xml_path = os.path.join(XML_DIR, equipment_model, 'bimanual_viperx_ee_insertion.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionEETask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif "sim_lifting_cube" in task_name:
        if equipment_model == "excavator_simple":
            xml_filename = 'single_viperx_ee_transfer_cube.xml'
        elif 'fairino5_single' in equipment_model:
            if 'with_complex_scene' in task_name:
                xml_filename = 'fairino_fr5_ee_lifting_cube_with_complex_scene.xml'
            else:
                xml_filename = 'fairino_fr5_ee_lifting_cube.xml'
        else:
            xml_filename = 'single_viperx_ee_transfer_cube.xml'
        xml_path = os.path.join(XML_DIR, equipment_model, xml_filename)
        physics = mujoco.Physics.from_xml_path(xml_path) # 加载物理引擎，其能够读取关节位置、推进一个step、渲染图像
        # Todo 临时增加一个挖掘机的class,后续合并进来
        use_complex_scene = 'with_complex_scene' in task_name
        if equipment_model == "excavator_simple":
            task = ExcavatorSimpleLiftingCubeEETask(random=False, arm_nums=1, equipment_model=equipment_model)
        else:
            task = LiftingCubeEETask(
                random=False,
                arm_nums=1,
                equipment_model=equipment_model,
                use_complex_scene=use_complex_scene,
            ) # 任务对象，初始化基本任务设置、奖励函数、观测数据提取
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env

class BimanualViperXEETask(base.Task):
    def __init__(self, random=None, arm_nums=2, equipment_model='vx300s_single'):
        super().__init__(random=random)
        self.arm_nums = arm_nums  # 2 双臂, 1 单臂
        self.equipment_model = equipment_model

    # 如果是挖掘机模型，返回True
    def _is_excavator(self):
        return 'excavator_simple' in self.equipment_model

    # 根据joint_name获取qpos中的索引
    @staticmethod
    def _get_joint_qpos_index(physics, joint_name):
        joint_id = physics.model.name2id(joint_name, 'joint')
        return int(physics.model.jnt_qposadr[joint_id])

    # 根据joint_name获取qvel中的索引
    @staticmethod
    def _get_joint_qvel_index(physics, joint_name):
        joint_id = physics.model.name2id(joint_name, 'joint')
        return int(physics.model.jnt_dofadr[joint_id])

    # 获取挖掘机主要关节的qpos，返回一个包含主要关节位置的numpy数组
    def _get_excavator_qpos(self, physics):
        qpos_raw = physics.data.qpos
        qpos_indices = [self._get_joint_qpos_index(physics, joint_name) for joint_name in EXCAVATOR_MAIN_JOINTS]
        return qpos_raw[qpos_indices].copy()

    # 获取挖掘机主要关节的qvel，返回一个包含主要关节速度的numpy数组
    def _get_excavator_qvel(self, physics):
        qvel_raw = physics.data.qvel
        qvel_indices = [self._get_joint_qvel_index(physics, joint_name) for joint_name in EXCAVATOR_MAIN_JOINTS]
        return qvel_raw[qvel_indices].copy()

    # 在物理仿真之前执行，physics 是 mujoco.Physics 类的实例，是 MuJoCo 物理引擎的 Python 接口，包含了整个仿真世界的所有状态和配置。
    def before_step(self, action, physics):
        # 分别拿到左和右双臂的xyz、四元数和夹爪状态
        if self.arm_nums == 2:
            a_len = len(action) // 2
            action_left = action[:a_len]
            action_right = action[a_len:]

            # set mocap position and quat，设置mocap的位置和姿态，mujoco中能够通过mocap控制末端执行器的位置和姿态
            # left，将action_left拷贝到physics.data.mocap_pos[0]
            # physics包含model和data：
            # model, 从xml中加载，主要包含静态配置
            # data, 动态状态，每步更新（mjdata）

            # left
            np.copyto(physics.data.mocap_pos[0], action_left[:3])
            np.copyto(physics.data.mocap_quat[0], action_left[3:7])
            # right
            np.copyto(physics.data.mocap_pos[1], action_right[:3])
            np.copyto(physics.data.mocap_quat[1], action_right[3:7])

            # set gripper
            g_left_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_left[7])
            g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[7])
            # ctrl[0] = g_left_ctrl     # 左夹爪手指1（向内）
            # ctrl[1] = -g_left_ctrl    # 左夹爪手指2（向外，方向相反）
            # ctrl[2] = g_right_ctrl    # 右夹爪手指1
            # ctrl[3] = -g_right_ctrl   # 右夹爪手指2（方向相反）
            np.copyto(physics.data.ctrl, np.array([g_left_ctrl, -g_left_ctrl, g_right_ctrl, -g_right_ctrl]))
        elif self.arm_nums == 1:
            action_left = None
            action_right = action

            if self._is_excavator():
                if len(action_right) == len(EXCAVATOR_MAIN_JOINTS):
                    np.copyto(physics.data.ctrl, action_right)
                else:
                    np.copyto(physics.data.mocap_pos[0], action_right[:3])
                    np.copyto(physics.data.mocap_quat[0], action_right[3:7])
            else:
                # 单臂认为是右臂
                np.copyto(physics.data.mocap_pos[0], action_right[:3])
                np.copyto(physics.data.mocap_quat[0], action_right[3:7])

                g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[7])
                np.copyto(physics.data.ctrl, np.array([g_right_ctrl, -g_right_ctrl]))
        else:
            raise NotImplementedError
      
    def initialize_robots(self, physics):
        # 通过初始化mocap的位姿控制设备初始位置
        if self.arm_nums == 2:
            # reset joint position
            physics.named.data.qpos[:16] = START_ARM_POSE

            # reset mocap to align with end effector
            # to obtain these numbers:
            # (1) make an ee_sim env and reset to the same start_pose
            # (2) get env._physics.named.data.xpos['vx300s_left/gripper_link']
            #     get env._physics.named.data.xquat['vx300s_left/gripper_link']
            #     repeat the same for right side
            np.copyto(physics.data.mocap_pos[0], [-0.31718881, 0.5, 0.29525084])
            np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
            # right
            np.copyto(physics.data.mocap_pos[1], np.array([0.31718881, 0.49999888, 0.29525084]))
            np.copyto(physics.data.mocap_quat[1],  [1, 0, 0, 0])

            # reset gripper control
            close_gripper_control = np.array([
                PUPPET_GRIPPER_POSITION_CLOSE,
                -PUPPET_GRIPPER_POSITION_CLOSE,
                PUPPET_GRIPPER_POSITION_CLOSE,
                -PUPPET_GRIPPER_POSITION_CLOSE,
            ])
            np.copyto(physics.data.ctrl, close_gripper_control)

        elif self.arm_nums == 1:
            if "vx300s_single" in self.equipment_model:
                physics.named.data.qpos[:8] = START_SINGLE_ARM_POSE
                physics.forward()
                np.copyto(physics.data.mocap_pos[0], np.array([-0.095, 0.50, 0.425]))
                np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
            elif "fairino5_single" in self.equipment_model:
                physics.named.data.qpos[:8] = START_FAIRINO_POSE
                physics.forward()
                # np.copyto(physics.data.mocap_pos[0], np.array([0, 0.396, 0.38]))
                # np.copyto(physics.data.mocap_quat[0], [0.707, 0, 0.707, 0])

                tcp_pos = physics.named.data.site_xpos['tcp_center'].copy()
                tcp_mat = physics.named.data.site_xmat['tcp_center'].copy()

                # 将旋转矩阵 (xmat) 转换为 MuJoCo 标准的四元数 [w, x, y, z]
                tcp_quat = np.zeros(4)
                mujoco.mju_mat2Quat(tcp_quat, tcp_mat.reshape(9))

                # 将 Mocap 吸附到夹爪最尖端的 TCP 中心
                np.copyto(physics.data.mocap_pos[0], tcp_pos)
                np.copyto(physics.data.mocap_quat[0], tcp_quat)
            elif "excavator_simple" in self.equipment_model:
                for joint_name, joint_value in zip(EXCAVATOR_MAIN_JOINTS, EXCAVATOR_START_POSE):
                    physics.named.data.qpos[joint_name] = joint_value
                physics.forward()
                np.copyto(physics.data.mocap_pos[0], np.array([4.8, 0.0, 0.45]))
                np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
            else:
                raise NotImplementedError(f"Unknown equipment model: {self.equipment_model}")
            
            # 为挖掘机做特殊操作 
            if self._is_excavator():
                np.copyto(physics.data.ctrl, EXCAVATOR_START_POSE)
            else:
                close_gripper_control = np.array([
                    PUPPET_GRIPPER_POSITION_CLOSE,
                    -PUPPET_GRIPPER_POSITION_CLOSE,
                ])
                np.copyto(physics.data.ctrl, close_gripper_control)
        else:
            raise NotImplementedError(f"Unknown arm_nums: {self.arm_nums}")

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    def get_qpos(self, physics):
        qpos_raw = physics.data.qpos.copy()
        if self.arm_nums == 2:
            left_qpos_raw = qpos_raw[:8]
            right_qpos_raw = qpos_raw[8:16]
            left_arm_qpos = left_qpos_raw[:6]
            right_arm_qpos = right_qpos_raw[:6]
            left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
            right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
            return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])
        elif self.arm_nums == 1:
            # 实际上对于挖掘机来说，其没有夹爪
            if self._is_excavator():
                return self._get_excavator_qpos(physics)
            right_qpos_raw = qpos_raw[:8]
            right_arm_qpos = right_qpos_raw[:6]
            right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
            return np.concatenate([right_arm_qpos, right_gripper_qpos])
        else:
            raise NotImplementedError

    def get_qvel(self, physics):
        qvel_raw = physics.data.qvel.copy()
        if self.arm_nums == 2:
            left_qvel_raw = qvel_raw[:8]
            right_qvel_raw = qvel_raw[8:16]
            left_arm_qvel = left_qvel_raw[:6]
            right_arm_qvel = right_qvel_raw[:6]
            left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
            right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
            return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])
        elif self.arm_nums == 1:
            if self._is_excavator():
                return self._get_excavator_qvel(physics)
            right_qvel_raw = qvel_raw[:8]
            right_arm_qvel = right_qvel_raw[:6]
            right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
            return np.concatenate([right_arm_qvel, right_gripper_qvel])
        else:
            raise NotImplementedError

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
        obs['images']['cockpit'] = physics.render(height=480, width=640, camera_id='cockpit')
        # obs['images']['right_pillar'] = physics.render(height=480, width=640, camera_id='right_pillar')
        if self.arm_nums == 2:
            # used in scripted policy to obtain starting pose
            obs['mocap_pose_left'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
            obs['mocap_pose_right'] = np.concatenate([physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()
        elif self.arm_nums == 1:
            obs['mocap_pose_right'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        else:
            raise NotImplementedError

        # used when replaying joint trajectory
        obs['gripper_ctrl'] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        raise NotImplementedError


class TransferCubeEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        cube_pose = sample_box_pose()
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        # 判断夹爪和盒子的接触情况，如果夹爪接触盒子，则touch_left_gripper/touch_right_gripper为True
        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        # 判断盒子和桌面的接触情况，如果盒子接触桌面，则touch_table为True
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted 表示通过右夹爪抬起盒子
            reward = 2
        if touch_left_gripper: # attempted transfer 表示左夹爪接触盒子
            reward = 3
        if touch_left_gripper and not touch_table: # successful transfer 表示左夹爪抬起盒子，成功转移盒子
            reward = 4
        return reward

class ExcavatorSimpleLiftingCubeEETask(BimanualViperXEETask):
    def __init__(self, random=None, arm_nums=1, equipment_model='vx300s_single'):
        super().__init__(random=random, arm_nums=arm_nums, equipment_model=equipment_model)
        self.max_reward = 4
        self._touched_box = False # 铲斗是否碰触过box

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        self._touched_box = False
        # randomize box position
        cube_pose = sample_box_pose_for_excavator()
        box_start_idx = physics.model.name2id('red_box_joint', 'joint') # 根据名字找到box的索引
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        # 获取箱子状态，在该任务下，qpos前8维是机械臂状态，后面的是箱子状态
        # env_state = physics.data.qpos.copy()[8:] # 备份
        box_joint_id = physics.model.name2id('red_box_joint', 'joint')
        box_start_idx = physics.model.jnt_qposadr[box_joint_id]
        env_state = physics.data.qpos.copy()[box_start_idx:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        # physics.data.ncon 获取当前ts的接触数量，遍历每一个接触对，拿到存在接触的geom名称
        for i_contact in range(physics.data.ncon):
            # i_contact为一个接触对对应的id，通过id拿到两个相互接触的geom id
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            # === debug ====
            # if name_geom_1 == "red_box" or name_geom_2 == "red_box":
            #     print(f"contact pair: {name_geom_1}, {name_geom_2}")
            # if name_geom_1 == "yellow_tray" or name_geom_2 == "yellow_tray":
            #     print(f"contact pair: {name_geom_1}, {name_geom_2}")
            # === debug ====
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        box_geom = "red_box"
        bucket_geom = "excavator_bucket"
        tray_geom = "yellow_dump_tray"

        touch_bucket_box = (box_geom, bucket_geom) in all_contact_pairs or (bucket_geom, box_geom) in all_contact_pairs
        touch_bucket_tray = (bucket_geom, tray_geom) in all_contact_pairs or (tray_geom, bucket_geom) in all_contact_pairs

        reward = 0
        if touch_bucket_box:
            self._touched_box = True
            reward = 1
        if self._touched_box and not touch_bucket_box:
            reward = 2
        if touch_bucket_tray:
            reward = 4
        return reward

class LiftingCubeEETask(BimanualViperXEETask):
    def __init__(self, random=None, arm_nums=1, equipment_model='vx300s_single', use_complex_scene=False):
        super().__init__(random=random, arm_nums=arm_nums, equipment_model=equipment_model)
        self.max_reward = 4
        # False: 原始单目标方块场景（只随机 red_box）
        # True : 复杂场景（随机 red_box + 3 个干扰方块）
        self.use_complex_scene = use_complex_scene
        # 默认为 red_box
        self.current_target_id = 0
        self.target_geom_name = 'red_box'
        # 可选：用于反事实数据生成时固定布局（shape=(28,)）
        self.fixed_scene_pose = None

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # 先把机器人本体复位到标准初始姿态
        self.initialize_robots(physics)
        # 再随机场景中物体的位置（而不是使用 XML 中写死的位置）
        if self.use_complex_scene:
            # flat_scene_pose shape = (28,) = 4 * (xyz + quat)
            # 固定顺序:
            # [red_box, distractor_box_1, distractor_box_2, distractor_box_3]
            if self.fixed_scene_pose is None:
                flat_scene_pose = sample_complex_scene_pose()
            else:
                flat_scene_pose = np.asarray(self.fixed_scene_pose, dtype=np.float64).reshape(-1)
                if flat_scene_pose.size != 28:
                    raise ValueError(
                        f'Unexpected fixed_scene_pose size={flat_scene_pose.size}, expected 28'
                    )
            joint_names = [
                'red_box_joint',
                'distractor_box_1_joint',
                'distractor_box_2_joint',
                'distractor_box_3_joint',
            ]
            for i, joint_name in enumerate(joint_names):
                # 每个物体占 7 维: [x, y, z, qw, qx, qy, qz]
                pose = flat_scene_pose[i * 7:(i + 1) * 7]
                try:
                    joint_id = physics.model.name2id(joint_name, 'joint')
                except (KeyError, ValueError):
                    # 某些 XML 可能不包含全部干扰物，允许跳过非目标物体
                    continue
                # name2id 返回的是 joint id，不是 qpos 下标；
                # 需要通过 jnt_qposadr 映射到 qpos 起始索引
                joint_qpos_start = int(physics.model.jnt_qposadr[joint_id])
                np.copyto(physics.data.qpos[joint_qpos_start:joint_qpos_start + 7], pose)
        else:
            # 原始任务：只随机目标方块 red_box
            cube_pose = sample_box_pose()
            box_joint_id = physics.model.name2id('red_box_joint', 'joint')
            box_start_idx = int(physics.model.jnt_qposadr[box_joint_id]) # 根据名字找到box在qpos中的起始索引
            np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        # print(f"randomized object pose(s)")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        # 获取箱子状态，在该任务下，qpos前8维是机械臂状态，后面的是箱子状态
        env_state = physics.data.qpos.copy()[8:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        # physics.data.ncon 获取当前ts的接触数量，遍历每一个接触对，拿到存在接触的geom名称
        for i_contact in range(physics.data.ncon):
            # i_contact为一个接触对对应的id，通过id拿到两个相互接触的geom id
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            # === debug ====
            # if name_geom_1 == "red_box" or name_geom_2 == "red_box":
            #     print(f"contact pair: {name_geom_1}, {name_geom_2}")
            # if name_geom_1 == "yellow_tray" or name_geom_2 == "yellow_tray":
            #     print(f"contact pair: {name_geom_1}, {name_geom_2}")
            # === debug ====
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        target_geom = getattr(self, 'target_geom_name', 'red_box')
        right_finger_geom = "vx300s_right/10_right_gripper_finger"
        left_finger_geom  = "vx300s_right/10_left_gripper_finger"
        table_geom = "table"
        tray_geom = "yellow_tray"

        # 判断夹爪与目标方块接触（左右任一手指接触即可）
        touch_left_gripper = (
            (target_geom, left_finger_geom) in all_contact_pairs
            or (left_finger_geom, target_geom) in all_contact_pairs
        )
        touch_right_gripper = (
            (target_geom, right_finger_geom) in all_contact_pairs
            or (right_finger_geom, target_geom) in all_contact_pairs
        )
        touch_gripper = touch_left_gripper or touch_right_gripper
        # 判断目标方块与桌面/托盘接触
        touch_table = (
            (target_geom, table_geom) in all_contact_pairs
            or (table_geom, target_geom) in all_contact_pairs
        )
        touch_tray = (
            (target_geom, tray_geom) in all_contact_pairs
            or (tray_geom, target_geom) in all_contact_pairs
        )

        reward = 0
        if touch_gripper:
            reward = 1
        if touch_gripper and not touch_table:
            reward = 2
        if touch_gripper and touch_tray:
            reward = 3
        if not touch_gripper and touch_tray:
            reward = 4
        return reward

class InsertionEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize peg and socket position
        peg_pose, socket_pose = sample_insertion_pose()
        id2index = lambda j_id: 16 + (j_id - 16) * 7 # first 16 is robot qpos, 7 is pose dim # hacky

        peg_start_id = physics.model.name2id('red_peg_joint', 'joint')
        peg_start_idx = id2index(peg_start_id)
        np.copyto(physics.data.qpos[peg_start_idx : peg_start_idx + 7], peg_pose)
        # print(f"randomized cube position to {cube_position}")

        socket_start_id = physics.model.name2id('blue_socket_joint', 'joint')
        socket_start_idx = id2index(socket_start_id)
        np.copyto(physics.data.qpos[socket_start_idx : socket_start_idx + 7], socket_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper = ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = ("socket-1", "table") in all_contact_pairs or \
                             ("socket-2", "table") in all_contact_pairs or \
                             ("socket-3", "table") in all_contact_pairs or \
                             ("socket-4", "table") in all_contact_pairs
        peg_touch_socket = ("red_peg", "socket-1") in all_contact_pairs or \
                           ("red_peg", "socket-2") in all_contact_pairs or \
                           ("red_peg", "socket-3") in all_contact_pairs or \
                           ("red_peg", "socket-4") in all_contact_pairs
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table): # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table): # peg and socket touching
            reward = 3
        if pin_touched: # successful insertion
            reward = 4
        return reward
