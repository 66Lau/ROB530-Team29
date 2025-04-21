import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_csv_data(file_path):
    data = pd.read_csv(file_path)  # 加载 CSV 文件
    return data.values  # 转换为 NumPy 数组，形状为 [num_steps, num_experiments]


    return mean + noise

def ema_filter(data, alpha=0.1):
    ema_data = np.zeros_like(data)
    ema_data[0] = data[0]
    for t in range(1, len(data)):
        ema_data[t] = alpha * data[t] + (1 - alpha) * ema_data[t - 1]
    return ema_data


baseline_files = ["../data/dynamics_data/mode1/recon_loss_mode1_1.csv", 
                "../data/dynamics_data/mode1/recon_loss_mode1_2.csv",
                "../data/dynamics_data/mode1/recon_loss_mode1_3.csv",
                "../data/dynamics_data/mode2/recon_loss_mode2_1.csv", 
                "../data/dynamics_data/mode2/recon_loss_mode2_2.csv",
                "../data/dynamics_data/mode2/recon_loss_mode2_3.csv",
                "../data/dynamics_data/mode3/recon_loss_mode3_1.csv",
                "../data/dynamics_data/mode3/recon_loss_mode3_2.csv",
                "../data/dynamics_data/mode3/recon_loss_mode3_3.csv",
                "../data/dynamics_data/mode4/recon_loss_mode4_1.csv",
                "../data/dynamics_data/mode4/recon_loss_mode4_2.csv",
                "../data/dynamics_data/mode4/recon_loss_mode4_3.csv"]


orin_data = [load_csv_data(file) for file in baseline_files]
all_data = []

steps = min([data.shape[0] for data in orin_data])-1
mode = 4

for i, data in enumerate(orin_data):
    all_data.append(data[0:steps, 2])

print( np.array(all_data).shape)

all_data = np.array(all_data).reshape(4, -1, steps)

# 合并数据到一个列表
# all_data = [data_baseline1, data_baseline2, data_baseline3]
labels = ["max_mode = 1", "max_mode = 2", "max_mode = 3", "max_mode = 4"]
colors = ["green", "purple", "blue", "orange"]

# 绘图
plt.figure(figsize=(10, 6))

for i in range(mode):
    max_vals = np.max(all_data[i,:,:], axis=0)
    min_vals = np.min(all_data[i,:,:], axis=0)
    mean = np.mean(all_data[i,:,:], axis=0)
    std = np.std(all_data[i,:,:], axis=0)
    x = np.arange(steps)

    mean = ema_filter(mean, alpha=0.1)
    std = ema_filter(std, alpha=0.01)

    # plt.plot(x, smoothed_data, label=labels[i], color=colors[i], alpha=0.8)

    # window_size = 100  # 滑动窗口大小
    # padding = window_size // 2
    # padded_data = np.pad(data, (padding, padding), mode='edge')  # 填充数据以处理边界
    # upper_bound = np.array([np.max(padded_data[i:i + window_size]) for i in range(len(data))])
    # lower_bound = np.array([np.min(padded_data[i:i + window_size]) for i in range(len(data))])
    # plt.fill_between(x, lower_bound, upper_bound, color=colors[i], alpha=0.1)

    
    # 绘制均值曲线
    plt.plot(x, mean, label=labels[i], color=colors[i], alpha=0.8)
    
    # 绘制置信区间 (标准误差)
    plt.fill_between(x, mean - std, mean + std, color=colors[i], alpha=0.2)
    # plt.fill_between(x, min_vals, max_vals, color=colors[i], alpha=0.2)

# 设置图例和标签
plt.ylim(0, 2)
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Dynamics Prediction Loss", fontsize=14)
# plt.title("Training Curves with Confidence Intervals", fontsize=16)
plt.legend(fontsize=12)
plt.xticks(fontsize=14)  
plt.yticks(fontsize=14) 
plt.grid(True)


# 保存或显示图像
plt.savefig("../data/Dynamics Prediction Loss.png", dpi=300, bbox_inches='tight')
plt.show()
