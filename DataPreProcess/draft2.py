import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Example: Simulated dataset (n observations, m features)
n, m = 100, 5  # 100 samples, 5 features
data = np.random.rand(n, m)

# Example: Clustering output from your algorithm
labels = np.random.randint(0, 3, size=n)  # Assume 3 clusters

# Reduce dimensionality to 2D using PCA
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)

# Scatter plot with cluster colors
plt.figure(figsize=(8, 6))
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.colorbar(label="Cluster ID")
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('2D Visualization of Clusters')
plt.show()