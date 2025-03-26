import pickle
import numpy as np


data = pickle.load(open("/data2/lzixuan/furniture-bench/scripted_sim_demo/one_leg/2025-02-19-03:51:51/2025-02-19-03:51:51.pkl", "rb"))

concat_order = ["ee_pos", "ee_quat", "ee_pos_vel", "ee_ori_vel", "joint_positions", "joint_velocities", "joint_torques", "gripper_width"]

states = np.stack([np.concatenate([data["observations"][i]["robot_state"][k].reshape(-1) for k in concat_order], axis=-1) for i in range(len(data["observations"]))], axis=0)

print(states.shape)

print(type(data["skills"][0]))

print(data["actor_id_to_name"])