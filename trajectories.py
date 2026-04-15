import numpy as np
from pyquaternion import Quaternion


def _build_sim_transfer_cube(ts):
    init_mocap_pose_right = ts.observation['mocap_pose_right']
    init_mocap_pose_left = ts.observation['mocap_pose_left']

    box_info = np.array(ts.observation['env_state'])
    box_xyz = box_info[:3]

    gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
    gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

    meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)
    meet_xyz = np.array([0, 0.5, 0.25])

    left_trajectory = [
        {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0, "desc": "初始位置"},
        {"t": 100, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1, "desc": "接近交接点"},
        {"t": 260, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1, "desc": "到达交接点"},
        {"t": 310, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 0, "desc": "闭合夹爪"},
        {"t": 360, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0, "desc": "后退"},
        {"t": 400, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0, "desc": "保持"},
    ]

    right_trajectory = [
        {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0, "desc": "初始位置"},
        {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1, "desc": "箱子上方"},
        {"t": 130, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 1, "desc": "下降"},
        {"t": 170, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 0, "desc": "闭合夹爪"},
        {"t": 200, "xyz": meet_xyz + np.array([0.05, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 0, "desc": "接近交接点"},
        {"t": 220, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 0, "desc": "到达交接点"},
        {"t": 310, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1, "desc": "打开夹爪"},
        {"t": 360, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1, "desc": "后退"},
        {"t": 400, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1, "desc": "保持"},
    ]

    meta = {
        "box_xyz": box_xyz,
        "meet_xyz": meet_xyz,
    }

    return left_trajectory, right_trajectory, meta

def _build_sim_lifting_cube(ts):
    init_mocap_pose_right = ts.observation['mocap_pose_right']

    box_info = np.array(ts.observation['env_state'])
    box_xyz = box_info[:3]

    gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
    gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

    meet_xyz = np.array([0, 0.5, 0.25])

    right_trajectory = [
        {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0, "desc": "初始位置"},
        {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1, "desc": "箱子上方"},
        {"t": 130, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 1, "desc": "下降"},
        {"t": 170, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 0, "desc": "闭合夹爪"},
        {"t": 200, "xyz": meet_xyz + np.array([0.05, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 0, "desc": "接近交接点"},
        {"t": 220, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 0, "desc": "到达交接点"},
        {"t": 310, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1, "desc": "打开夹爪"},
        {"t": 360, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1, "desc": "后退"},
        {"t": 400, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1, "desc": "保持"},
    ]

    meta = {
        "box_xyz": box_xyz,
        "meet_xyz": meet_xyz,
    }

    return None,right_trajectory, meta

def get_trajectory(task_name, ts):
    if task_name == 'sim_transfer_cube_scripted':
        return _build_sim_transfer_cube(ts)
    elif 'sim_lifting_cube' in task_name:
        return _build_sim_lifting_cube(ts)

    raise ValueError(f"Unknown task name: {task_name}")
