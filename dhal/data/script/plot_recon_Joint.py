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


def extract_info(data_list, key="representation"):
    representations = [step[key] for step in data_list]
    array = np.vstack(representations)
    return array

def plot_joint_comparison(joint_pos, recon_joint_pos, mode, save_path=None, title=None):
    step, n = joint_pos.shape
    time_steps = np.arange(joint_pos.shape[0])
    colors = plt.cm.viridis(np.linspace(0, 1, 6))  # Use 6 colors for the 6 joints

    plt.figure(figsize=(14, 6))
    for i in range(n):
        if i == 0 :
            label = "FR_H(real)"
            label_recon = "FR_H(predict)"
        elif i == 1 :
            label = "FR_T(real)"
            label_recon = "FR_T(predict)"
        elif i == 2 :
            label = "FR_C(real)"
            label_recon = "FR_C(predict)"
        plt.plot(time_steps, joint_pos[:, i], label=label, color=colors[i], linewidth=1.5)
        plt.plot(time_steps, recon_joint_pos[:, i], linestyle='--', label=label_recon, color=colors[i], linewidth=1.5)

    mode_colors = ["#F08080", "#336699", "#CCFF66"]  # 高对比且协调的颜色


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
                alpha=0.1,  # Adjust alpha for better contrast
                # label=f"Mode {i + 1}"
            )


    plt.xlabel("Time Steps", fontsize=20)
    plt.ylabel("Joint Position", fontsize=20)
    # plt.title(title)
    plt.legend(fontsize=12,loc='upper right')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

file_path = "/home/lau/gym/go1_gym/data/data/log_recon_0124_2.pkl"
with open(file_path, 'rb') as file:
    data = pickle.load(file)

data = data['hardware_closed_loop'][1]
print("\nData Keys:", data[0].keys(),"\n")

joint_pos = extract_info(data, key="joint_pos")
mode = extract_info(data, key="mode")

default_dof_pos = np.array([-0.1,  1.0, -1.8,
                              0.1,  1.5, -2.4,
                             -0.1,  1.0, -1.8,
                              0.1,  1.5, -2.4])

recon_joint_pos = extract_info(data, key="recon_x")[:, 8:20] + default_dof_pos.reshape(1, 12)

# Plot the first 6 dimensions
plot_joint_comparison(joint_pos[:, :3], recon_joint_pos[:, :3], mode, save_path="../data/Joint_pred.png", title="Joint Position Comparison")


