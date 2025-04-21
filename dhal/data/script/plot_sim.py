import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA, PCA
from scipy.io import savemat, loadmat
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd
from matplotlib.colors import to_rgba


def extract_info(data_list, key = "representation"):
    representations = [step[key] for step in data_list]
    array = np.vstack(representations)
    return array

def plot_time_series(result_array, save_path=None, fill=False, linewidth=1, title=None):
    i, n = result_array.shape
    time_steps = np.arange(i)
    colors = plt.cm.viridis(np.linspace(0, 1, n))

    plt.figure(figsize=(10, 6))
    for j in range(n):
        plt.plot(time_steps, result_array[:, j], label=f"Dimension {j+1}", color=colors[j], linewidth=linewidth)
        if fill:
            plt.fill_between(time_steps, result_array[:, j], alpha=0.3, color=colors[j]) 
    
    plt.xlabel("Time Steps")
    plt.ylabel("Values")
    plt.title(title)
    # plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(save_path)

def plot_joint_with_mode_background(joint_pos, mode, save_path=None):
    time_steps = np.arange(joint_pos.shape[0])
    joint_colors = plt.cm.viridis(np.linspace(0, 1, joint_pos.shape[1]))
    mode_colors = ["#F08080", "#00FA9A", "#00FFFF"]  # 高对比且协调的颜色

    # Initialize figure
    plt.figure(figsize=(12, 6))

    # Plot mode as background color
    for i in range(mode.shape[1]):
        mode_intervals = np.where(mode[:, i] == 1)[0]
        if len(mode_intervals) > 0:
            plt.fill_between(
                time_steps,
                -3,  # Extend below the minimum of joint_pos
                3,   # Extend above the maximum of joint_pos
                where=np.isin(time_steps, mode_intervals),
                color=mode_colors[i % len(mode_colors)],
                alpha=0.2,  # Adjust alpha for better contrast
                label=f"Mode {i + 1}"
            )

    # Plot joint positions
    for j in range(joint_pos.shape[1]):
        plt.plot(time_steps, joint_pos[:, j], color=joint_colors[j], linewidth=0.8, label=f"Joint {j + 1}")

    # Add labels, title, and legend
    plt.xlabel("Time Steps")
    plt.ylabel("Joint Position Values")
    plt.title("Joint Positions with Mode as Background")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")

    # Save and show the plot
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()



file_path = "/home/lau/gym/go1_gym/data/data/log9.pkl"
with open(file_path, 'rb') as file:
    data = pickle.load(file)

data = data['hardware_closed_loop'][1]
print("\nData Keys:", data[0].keys(),"\n")
num_steps = len(data)

joint_pos = extract_info(data, key = "joint_pos")
joint_vel = extract_info(data, key = "joint_vel")


# plot_time_series(joint_pos[:,0].reshape(-1,1), "../data/plot_joint_vel.png", title="Joint Velocity")
# plot_time_series(joint_vel[200:250,2].reshape(-1,1), "../data/plot_joint_vel.png", title="Joint Velocity")
# joint_pos = joint_pos[700:800,2]
# joint_vel = joint_vel[700:800,2]

indices = np.s_[700:800, 1000:1200,1400:1600]  # 定义切片范围
joint_pos = joint_pos[np.r_[indices], 2]
joint_vel = joint_vel[np.r_[indices], 2]

plt.figure(figsize=(12, 6), )

for i in range(10, 0, -1):  # 从透明到深色
    plt.plot(
        joint_pos, 
        joint_vel, 
        linewidth=12 + i, 
        alpha=0.03 * i, 
        color=to_rgba("#FFC18A", alpha=0.1)
    )

plt.plot(joint_pos, joint_vel, linewidth=5, alpha = 0.8, color = "#F99634")

plt.xlabel("Joint Position")
plt.ylabel("Joint Velocity")
# plt.title(" Phase Portrait of Hybrid Dynamical System")
# plt.legend()
# plt.grid(True)
plt.show()
plt.savefig("../data/Phase.png")
