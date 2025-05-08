import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load user data
users_file = "ModifiedDataset/user_matrix.csv"
users_df = pd.read_csv(users_file)

usernames = users_df['username']
users = users_df.drop('username', axis=1).to_numpy()  # Convert to NumPy array

# Number of clusters
centers = 4

# --- Step 1: K-Means++ Initialization ---
def kmeans_plus_plus(users, k):
    centroids = [users[np.random.randint(users.shape[0])]]
    for _ in range(1, k):
        dist_sq = np.min([np.sum((users - c) ** 2, axis=1) for c in centroids], axis=0)
        prob = dist_sq / dist_sq.sum()
        chosen_idx = np.random.choice(users.shape[0], p=prob)
        centroids.append(users[chosen_idx])
    return np.array(centroids)

# --- Step 2: Masked Euclidean Distance ---
def masked_euclidean(u, v):
    mask = (u != 0) & (v != 0)
    if not np.any(mask):
        return np.inf
    return np.sqrt(np.sum((u[mask] - v[mask]) ** 2))


def masked_cosine(u, v):
    mask = (u != 0) & (v != 0)
    if not np.any(mask):
        return 1.0  # maximum dissimilarity
    u_masked = u[mask]
    v_masked = v[mask]
    
    numerator = np.dot(u_masked, v_masked)
    denominator = np.linalg.norm(u_masked) * np.linalg.norm(v_masked)
    
    if denominator == 0:
        return 1.0  # undefined, treat as max distance
    return 1 - (numerator / denominator)

def jaccard_distance(u, v):
    u_bin = u != 0
    v_bin = v != 0

    intersection = np.sum(u_bin & v_bin)
    union = np.sum(u_bin | v_bin)

    if union == 0:
        return 1.0  # no ratings at all
    return 1 - (intersection / union)


centroids = kmeans_plus_plus(users, centers)
distances = pd.DataFrame(0.0, index=usernames.to_list(), columns=[f"Cluster {i+1}" for i in range(centers)])

start = time.time()
max_iterations = 50
previous_clusters = None

for iteration in range(max_iterations):
    print(f"{iteration + 1} - Clustering cycle")

    # --- Step 3: Compute Masked Distances ---
    for i in range(centers):
        for j in range(users.shape[0]):
            # distances.iloc[j, i] = masked_euclidean(users[j], centroids[i]) # Uncomment for Euclidean distance
            # distances.iloc[j, i] = masked_cosine(users[j], centroids[i])  # Uncomment for cosine distance
            distances.iloc[j, i] = jaccard_distance(users[j], centroids[i])  # Uncomment for Jaccard distance

    cluster_assignments = distances.idxmin(axis=1)

    used_centroids = set()
    empty_clusters = []

    for m in range(centers):
        cluster_users = users[cluster_assignments == f"Cluster {m + 1}"]
        if len(cluster_users) > 0:
            new_centroid = np.zeros(users.shape[1])
            count = np.zeros(users.shape[1])
            for user in cluster_users:
                non_zero = user != 0
                new_centroid[non_zero] += user[non_zero]
                count[non_zero] += 1
            with np.errstate(divide='ignore', invalid='ignore'):
                avg = np.divide(new_centroid, count, where=count != 0)
                avg[count == 0] = 0
            centroids[m] = avg
            used_centroids.add(tuple(centroids[m]))
        else:
            empty_clusters.append(m)

    if empty_clusters:
        print(f"Empty clusters detected: {empty_clusters}")
        min_distances = np.min(
            [np.linalg.norm(users - centroids[c], axis=1) for c in range(centers)],
            axis=0
        )
        sorted_indices = np.argsort(min_distances)[::-1]
        for m in empty_clusters:
            for idx in sorted_indices:
                candidate = tuple(users[idx])
                if candidate not in used_centroids:
                    centroids[m] = users[idx]
                    used_centroids.add(candidate)
                    print(f"Reassigned Cluster {m+1} to user index {idx}")
                    break

    if iteration > 0 and previous_clusters.equals(cluster_assignments):
        print(f"Converged after {iteration + 1} iterations")
        break

    previous_clusters = cluster_assignments.copy()

end = time.time()
print(f"Time taken: {end - start:.2f} seconds")

# --- Save centroids and assignments ---
centroids_df = pd.DataFrame(centroids, columns=[f"Feature {i+1}" for i in range(users.shape[1])])
centroids_df.to_csv("ProcessedData/centroids.csv", index=False)
cluster_assignments.to_csv("ProcessedData/clusters.csv")

# --- Plot cluster distribution ---
cluster_counts = cluster_assignments.value_counts()
plt.figure(figsize=(8, 5))
cluster_counts.sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
plt.xlabel("Cluster")
plt.ylabel("Number of Users")
plt.title("User Distribution Across Clusters")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# --- Inertia Calculation ---
inertia = 0
for idx, cluster_label in enumerate(cluster_assignments):
    cluster_idx = int(cluster_label.split()[-1]) - 1
    user_vector = users[idx]
    centroid_vector = centroids[cluster_idx]
    dist = masked_euclidean(user_vector, centroid_vector)
    inertia += dist ** 2 if dist != np.inf else 0

print(f"Inertia: {inertia:.2f}")

# --- Silhouette Score ---
labels = cluster_assignments.str.extract('(\d+)').astype(int).values.flatten()
silhouette_avg = silhouette_score(users, labels)
print(f"Silhouette Score: {silhouette_avg:.4f}")

# --- PCA for Visualization ---
pca = PCA(n_components=2)
users_2d = pca.fit_transform(users)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(users_2d[:, 0], users_2d[:, 1], c=labels, cmap='tab10', s=30, alpha=0.7)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('User Clusters Visualized in 2D (PCA)')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
