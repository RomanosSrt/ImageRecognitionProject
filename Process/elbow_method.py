import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator  # install with: pip install kneed

# Load user matrix
df = pd.read_csv("ModifiedDataset/user_matrix.csv")
X = df.drop(columns='username').to_numpy()

# Compute distortion (inertia per sample)
distortions = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X)
    distortion = kmeans.inertia_ / X.shape[0]
    distortions.append(distortion)

# Find optimal K using the "knee" method
knee_locator = KneeLocator(k_range, distortions, curve='convex', direction='decreasing')
optimal_k = knee_locator.elbow

# Plot distortion curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, distortions, marker='o')
if optimal_k:
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K = {optimal_k}')
plt.title('Elbow Method using Distortion')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Average Distortion')
plt.grid(True)
plt.legend()
plt.show()

# Output the selected optimal K
print(f"\n✅ Most suitable number of clusters based on elbow method: K = {optimal_k}")
