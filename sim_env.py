import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from constants import DT, START_FAIRINO_POSE, XML_DIR, START_ARM_POSE, START_SINGLE_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

from constants import EXCAVATOR_MAIN_JOINTS,EXCAVATOR_START_POSE 

import IPython
e = IPython.embed

BOX_POSE = [None] # to be changed from outside

def make_sim_env(task_name, equipment_model: str = 'vx300s_bimanual'):
    """
    Environment for simulated robot bi-manual manipulation, with joint position control
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
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
    if 'sim_transfer_cube' in task_name:
        xml_path = os.path.join(XML_DIR, equipment_model, 'bimanual_viperx_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_insertion' in task_name:
        xml_path = os.path.join(XML_DIR, equipment_model, 'bimanual_viperx_insertion.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_lifting_cube' in task_name:
        if equipment_model == 'excavator_simple':
            xml_path = os.path.join(XML_DIR, equipment_model, 'single_viperx_transfer_cube.xml')
            physics = mujoco.Physics.from_xml_path(xml_path)
            task = ExcavatorSimpleLiftingCubeTask(random=False, equipment_model=equipment_model)
        else:
            xml_path = os.path.join(XML_DIR, equipment_model, 'single_viperx_transfer_cube.xml')
            physics = mujoco.Physics.from_xml_path(xml_path)
            task = LiftingCubeTask(random=False, equipment_model=equipment_model)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env

class BimanualViperXTask(base.Task):
    def __init__(self, random=None, arm_nums=2):
        super().__init__(random=random)
        self.arm_nums = arm_nums  # 2 双臂, 1 单臂

    def before_step(self, action, physics):
        if self.arm_nums == 2:
            left_arm_action = action[:6]
            right_arm_action = action[7:7+6]
            normalized_left_gripper_action = action[6]
            normalized_right_gripper_action = action[7+6]

            left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_left_gripper_action)
            right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_right_gripper_action)

            full_left_gripper_action = [left_gripper_action, -left_gripper_action]
            full_right_gripper_action = [right_gripper_action, -right_gripper_action]

            env_action = np.concatenate([left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action])
            super().before_step(env_action, physics)
        elif self.arm_nums == 1:
            right_arm_action = action[:6]
            normalized_right_gripper_action = action[6]

            right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_right_gripper_action)

            full_right_gripper_action = [right_gripper_action, -right_gripper_action]

            env_action = np.concatenate([right_arm_action, full_right_gripper_action])
            super().before_step(env_action, physics)
        return

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
            right_qpos_raw = qpos_raw[:8]
            right_arm_qpos = right_qpos_raw[:6]
            right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
            return  np.concatenate([right_arm_qpos, right_gripper_qpos])
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
            right_qvel_raw = qvel_raw[:8]
            right_arm_qvel = right_qvel_raw[:6]
            right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
            return  np.concatenate([right_arm_qvel, right_gripper_qvel])
        else:
            raise NotImplementedError

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
        # obs['images']['right_pillar'] = physics.render(height=480, width=640, camera_id='right_pillar')

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class TransferCubeTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
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

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_left_gripper: # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table: # successful transfer
            reward = 4
        return reward


class InsertionTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7*2:] = BOX_POSE[0] # two objects
            # print(f"{BOX_POSE=}")
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

class LiftingCubeTask(BimanualViperXTask):
    def __init__(self, random=None, equipment_model='vx300s_single'):
        super().__init__(random=random, arm_nums=1)
        self.max_reward = 4
        self.equipment_model = equipment_model

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position

         # 根据机器人类型选择初始姿态
        if 'fairino5_single' in self.equipment_model:
            start_pose = START_FAIRINO_POSE
            print("use Fairino FR5 config file")
        elif 'vx300s_single' in self.equipment_model:
            start_pose = START_SINGLE_ARM_POSE
            print("use VX300s config file")
        else:
            raise ValueError(f"Unknown equipment model: {self.equipment_model}")

        with physics.reset_context():
            physics.named.data.qpos[:8] = start_pose
            np.copyto(physics.data.ctrl, start_pose)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        physics.forward()
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[8:]
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

        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs or ("vx300s_right/10_right_gripper_finger", "red_box") in all_contact_pairs 
        
        # 判断盒子和桌面的接触情况，如果盒子接触桌面，则touch_table为True
        touch_table = ("red_box", "table") in all_contact_pairs or ("table", "red_box") in all_contact_pairs 
        touch_tray = ("red_box", "yellow_tray") in all_contact_pairs or ("yellow_tray", "red_box") in all_contact_pairs 

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table:
            reward = 2
        if touch_right_gripper and touch_tray:
            reward = 3
        if not touch_right_gripper and touch_tray:
            reward = 4
        return reward

class ExcavatorSimpleLiftingCubeTask(base.Task):
    def __init__(self, random=None, equipment_model='excavator_simple'):
        super().__init__(random=random)
        self.max_reward = 4
        self.equipment_model = equipment_model
        self._touched_box = False # 铲斗是否碰触过box

    def before_step(self, action, physics):
        if len(action) != len(EXCAVATOR_MAIN_JOINTS):
            raise ValueError(f"Excavator action must be {len(EXCAVATOR_MAIN_JOINTS)}D, got {len(action)}")
        np.copyto(physics.data.ctrl, action)

    def initialize_episode(self, physics):
        self._touched_box = False # 铲斗是否碰触过box
        with physics.reset_context():
            for joint_name, joint_value in zip(EXCAVATOR_MAIN_JOINTS, EXCAVATOR_START_POSE):
                physics.named.data.qpos[joint_name] = joint_value
            np.copyto(physics.data.ctrl, EXCAVATOR_START_POSE)
            assert BOX_POSE[0] is not None
            box_joint_id = physics.model.name2id('red_box_joint', 'joint')
            box_start_idx = physics.model.jnt_qposadr[box_joint_id]
            physics.named.data.qpos[box_start_idx:box_start_idx + 7] = BOX_POSE[0]
        physics.forward()
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        box_joint_id = physics.model.name2id('red_box_joint', 'joint')
        box_start_idx = physics.model.jnt_qposadr[box_joint_id]
        env_state = physics.data.qpos.copy()[box_start_idx:]
        return env_state

    def get_qpos(self, physics):
        qpos_raw = physics.data.qpos
        qpos_indices = [physics.model.jnt_qposadr[physics.model.name2id(joint_name, 'joint')]
                        for joint_name in EXCAVATOR_MAIN_JOINTS]
        return qpos_raw[qpos_indices].copy()

    def get_qvel(self, physics):
        qvel_raw = physics.data.qvel
        qvel_indices = [physics.model.jnt_dofadr[physics.model.name2id(joint_name, 'joint')]
                        for joint_name in EXCAVATOR_MAIN_JOINTS]
        return qvel_raw[qvel_indices].copy()

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
        return obs

    def get_reward(self, physics):
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
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

# for test
def get_action(master_bot_left, master_bot_right):
    action = np.zeros(14)
    # arm action
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7:7+6] = master_bot_right.dxl.joint_states.position[:6]
    # gripper action
    left_gripper_pos = master_bot_left.dxl.joint_states.position[7]
    right_gripper_pos = master_bot_right.dxl.joint_states.position[7]
    normalized_left_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(left_gripper_pos)
    normalized_right_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(right_gripper_pos)
    action[6] = normalized_left_pos
    action[7+6] = normalized_right_pos
    return action

def test_sim_teleop():
    """ Testing teleoperation in sim with ALOHA. Requires hardware and ALOHA repo to work. """
    from interbotix_xs_modules.arm import InterbotixManipulatorXS

    BOX_POSE[0] = [0.2, 0.5, 0.05, 1, 0, 0, 0]

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_right', init_node=False)

    # setup the environment
    env = make_sim_env('sim_transfer_cube')
    ts = env.reset()
    episode = [ts]
    # setup plotting
    ax = plt.subplot()
    plt_img = ax.imshow(ts.observation['images']['angle'])
    plt.ion()

    for t in range(1000):
        action = get_action(master_bot_left, master_bot_right)
        ts = env.step(action)
        episode.append(ts)

        plt_img.set_data(ts.observation['images']['angle'])
        plt.pause(0.02)


if __name__ == '__main__':
    test_sim_teleop()

