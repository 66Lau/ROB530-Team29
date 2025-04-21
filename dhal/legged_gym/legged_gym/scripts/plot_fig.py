import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 设置路径（或根据需要改成 argparse）
pose_files = {
    "Pure Propagation": "prop_pose.jsonl",
    "Contact-aided InEKF": "contact_aid_pose.jsonl",
    "InEKF (Kinematics Only)": "inekf_kin_pose.jsonl",
    "InEKF (Kinematics + Rough Vel)": "inekf_extra_vel_pose.jsonl",
    "Ground Truth": "gt_pose.jsonl"
}

colors = {
    "Pure Propagation": "tab:blue",
    "Contact-aided InEKF": "tab:orange",
    "InEKF (Kinematics Only)": "tab:green",
    "InEKF (Kinematics + Rough Vel)": "tab:red",
    "Ground Truth": "tab:purple"
}

def load_pose_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            pos = entry["position"]
            data.append((pos["x"], pos["y"], pos["z"]))
    return data

# 读取所有方法的数据
all_data = {label: load_pose_data(path) for label, path in pose_files.items()}

# --- Plot X, Y, Z over step ---
for i, axis in enumerate(['x', 'y', 'z']):
    plt.figure(figsize=(10, 5))
    for label, data in all_data.items():
        values = [pos[i] for pos in data]
        plt.plot(values, label=label, color=colors[label])
    plt.title(f'{axis.upper()} Position over Steps')
    plt.xlabel('Step')
    plt.ylabel(f'{axis.upper()} Position (m)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{axis}_over_time.png", dpi=300)
    plt.show()

# --- 3D Trajectory Plot ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
for label, data in all_data.items():
    xs, ys, zs = zip(*data)
    ax.plot(xs, ys, zs, label=label, color=colors[label])
ax.set_title('3D Trajectories')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.legend()
plt.tight_layout()
plt.savefig("trajectory_3d.png", dpi=300)
plt.show()
