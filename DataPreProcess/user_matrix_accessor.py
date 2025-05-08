import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import pandas as pd

# Load user matrix
users_df = pd.read_csv("ModifiedDataset/user_matrix.csv")
users = users_df.drop('username', axis=1).to_numpy()  # NumPy array

# Distance functions
def masked_euclidean(u, v):
    mask = (u != 0) & (v != 0)
    return np.sqrt(np.sum((u[mask] - v[mask]) ** 2)) if np.any(mask) else np.inf

def masked_cosine(u, v):
    mask = (u != 0) & (v != 0)
    if not np.any(mask): return 1.0
    u_m, v_m = u[mask], v[mask]
    denom = np.linalg.norm(u_m) * np.linalg.norm(v_m)
    return 1 - (np.dot(u_m, v_m) / denom) if denom else 1.0

def jaccard_distance(u, v):
    u_bin, v_bin = u != 0, v != 0
    inter, union = np.sum(u_bin & v_bin), np.sum(u_bin | v_bin)
    return 1 - (inter / union) if union else 1.0

# KMeans++ initialization
def kmeans_plus_plus(data, k):
    centroids = [data[np.random.randint(data.shape[0])]]
    for _ in range(1, k):
        dist_sq = np.min([np.sum((data - c) ** 2, axis=1) for c in centroids], axis=0)
        prob = dist_sq / dist_sq.sum()
        centroids.append(data[np.random.choice(data.shape[0], p=prob)])
    return np.array(centroids)

# Evaluate clustering with given distance metric
def evaluate_kmeans(users, distance_fn, k_range):
    inertia_scores = []
    silhouette_scores = []
    
    for k in k_range:
        centroids = kmeans_plus_plus(users, k)
        for _ in range(30):
            distances = np.array([[distance_fn(user, centroid) for centroid in centroids] for user in users])
            assignments = np.argmin(distances, axis=1)

            new_centroids = []
            for i in range(k):
                cluster_members = users[assignments == i]
                if len(cluster_members) > 0:
                    mask = cluster_members != 0
                    sums = cluster_members.sum(axis=0)
                    counts = mask.sum(axis=0)
                    avg = np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)
                    new_centroids.append(avg)
                else:
                    new_centroids.append(users[np.random.randint(users.shape[0])])
            new_centroids = np.array(new_centroids)
            if np.allclose(new_centroids, centroids):
                break
            centroids = new_centroids

        inertia = sum((distance_fn(users[i], centroids[assignments[i]]) ** 2
                       if distance_fn(users[i], centroids[assignments[i]]) != np.inf else 0)
                      for i in range(users.shape[0]))
        silhouette = silhouette_score(users, assignments) if len(set(assignments)) > 1 else -1
        inertia_scores.append(inertia)
        silhouette_scores.append(silhouette)

    return inertia_scores, silhouette_scores

from sklearn.neighbors import NearestNeighbors

def hopkins_statistic(X, sampling_size=0.05):
    from numpy.random import uniform, choice
    X = X[np.random.choice(X.shape[0], size=int(sampling_size * X.shape[0]), replace=False)]
    n, d = X.shape
    m = int(0.1 * n)  # Number of random points

    nbrs = NearestNeighbors(n_neighbors=1).fit(X)

    rand_X = uniform(np.min(X, axis=0), np.max(X, axis=0), (m, d))
    u_dist, _ = nbrs.kneighbors(rand_X)
    w_dist, _ = nbrs.kneighbors(X[np.random.choice(n, m, replace=False)])

    u_sum = np.sum(u_dist)
    w_sum = np.sum(w_dist)
    return u_sum / (u_sum + w_sum)



# Run for all distance metrics
k_values = list(range(2, 11))
results = {}
for name, func in [('euclidean', masked_euclidean), ('cosine', masked_cosine), ('jaccard', jaccard_distance)]:
    inertia, silhouette = evaluate_kmeans(users, func, k_values)
    results[name] = {'inertia': inertia, 'silhouette': silhouette}

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

for name, res in results.items():
    axs[0].plot(k_values, res['inertia'], marker='o', label=name)
    axs[1].plot(k_values, res['silhouette'], marker='o', label=name)

axs[0].set_title('Elbow Method (Inertia vs. k)')
axs[0].set_xlabel('Number of Clusters (k)')
axs[0].set_ylabel('Inertia')
axs[0].legend()
axs[0].grid(True)

axs[1].set_title('Silhouette Score vs. k')
axs[1].set_xlabel('Number of Clusters (k)')
axs[1].set_ylabel('Silhouette Score')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()


from sklearn.preprocessing import MinMaxScaler
users_scaled = MinMaxScaler().fit_transform(users)


hopkins_score = hopkins_statistic(users_scaled)
print(f"Hopkins statistic: {hopkins_score:.4f}")
