import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA, PCA
from scipy.io import savemat, loadmat
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd

def extract_representations(data_list, key = "representation"):
    representations = [step[key] for step in data_list]
    array = np.vstack(representations)
    return array

def pca_with_rbf(input_matrix, n_components=2, gamma=0.8):
    """
    Perform Kernel PCA with RBF kernel.
    
    Parameters:
    - input_matrix: np.array, shape (i, n), where i is the number of data points and n is the number of features.
    - n_components: int, number of components to project to (default=2).
    - gamma: float, kernel coefficient for RBF kernel (default=0.1).
    
    Returns:
    - transformed_data: np.array, shape (i, n_components), the data projected to the new space.
    - kpca: KernelPCA object (for further inspection if needed).
    """
    # Perform Kernel PCA
    kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=gamma)
    transformed_data = kpca.fit_transform(input_matrix)
    
    
    return transformed_data, kpca

def pca(input_matrix, n_components=2):
    """
    Perform ordinary PCA.

    Parameters:
    - input_matrix: np.array, shape (i, n), where i is the number of data points and n is the number of features.
    - n_components: int, number of components to project to (default=2).

    Returns:
    - transformed_data: np.array, shape (i, n_components), the data projected to the new space.
    - pca: PCA object (for further inspection if needed).
    """
    # Perform PCA
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(input_matrix)
    
    return transformed_data, pca

def get_covarience_egien(data):
    dimension = data.shape[1]
    batch = data.shape[0]
    data_tilt = (data - np.mean(data,axis=0))/dimension
    covarience = (1/batch) * data_tilt @ data_tilt.T
    eigenvalues, eigenvectors =  np.linalg.eig(covarience)
    print(eigenvalues)


file_path = "/home/lau/gym/go1_gym/data/data/sim_data/sim.pkl"
with open(file_path, 'rb') as file:
    data = pickle.load(file)

data = data
print("\nData Keys:", data[0].keys())
num_steps = len(data)
representation = extract_representations(data, key = "intermediate_output_2")
mode = extract_representations(data, key = "mode")

# PCA
# transformed_data, kpca_model = pca_with_rbf(representation, n_components=2, gamma=0.05)
# transformed_data, kpca_model = pca(representation, n_components=2)

category_indices = np.argmax(mode, axis=1) 

# 定义颜色映射
color_list = [ "#59A14F", "#4E79A7","#E15759"]
if len(color_list) < len(np.unique(category_indices)):
    raise ValueError("Not enough colors defined for the number of unique modes.")

# 为每个类别分配颜色
colors = [color_list[index] for index in category_indices]

# t-SNE
scaler = StandardScaler()
X_scaled = scaler.fit_transform(representation)
tsne = TSNE(n_components=2, perplexity=1.5, random_state=42)  # n_components 表示降维至 2 维
transformed_data = tsne.fit_transform(X_scaled)

# Plot the 2D transformation result (for visualization)
plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=colors, marker='o', alpha=0.7,s=8)
# plt.plot(transformed_data[0:500:, 0], transformed_data[0:500:, 1], linewidth=1)
plt.title("t-SNE of Hidden Layer Output")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.show()
plt.savefig("../data/plot_representation.png")
