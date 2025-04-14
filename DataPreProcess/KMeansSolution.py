import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# Load user data
users_file = "ModifiedDataset/user_matrix.csv"
users = pd.read_csv(users_file)

usernames = users['username']
users = users.drop('username', axis=1).to_numpy()  # Convert to NumPy array

# Number of clusters
centers = 4

# --- Step 1: K-Means++ Initialization ---
def kmeans_plus_plus(users, k):
    """Initialize centroids using the K-Means++ method."""
    centroids = []
    centroids.append(users[np.random.randint(users.shape[0])])  # Pick first randomly

    for _ in range(1, k):
        # Compute squared distances from existing centroids
        dist_sq = np.min([np.sum((users - c) ** 2, axis=1) for c in centroids], axis=0)
        prob = dist_sq / dist_sq.sum()  # Probability distribution
        chosen_idx = np.random.choice(users.shape[0], p=prob)
        centroids.append(users[chosen_idx])

    return np.array(centroids).T  # Transpose to match expected shape

centroids = kmeans_plus_plus(users, centers)

# Initialize storage for distances and cluster assignments
distances = pd.DataFrame(0.0, index=usernames.to_list(), columns=[f"Cluster {i+1}" for i in range(centers)])

start = time.time()
max_iterations = 50
previous_clusters = None

for iteration in range(max_iterations):
    print(f"{iteration+1} - Clustering cycle")

    # --- Step 2: Compute Distances ---
    for i in range(centers):
        distances.iloc[:, i] = np.sqrt(np.sum((users - centroids[:, i].T) ** 2, axis=1))

    # Replace zero distances with NaN to avoid fake close assignments
    distances.replace(0, np.nan, inplace=True)

    # --- Step 3: Assign Each User to the Nearest Cluster ---
    cluster_assignments = distances.idxmin(axis=1)

    # Track assigned centroids to avoid duplicates
    used_centroids = set()

    empty_clusters = []

    for m in range(centers):
        cluster_users = users[cluster_assignments == f"Cluster {m + 1}"]

        if len(cluster_users) > 0:
            centroids[:, m] = cluster_users.mean(axis=0)
            used_centroids.add(tuple(centroids[:, m]))  # Track assigned centroid
        else:
            empty_clusters.append(m)  # Track empty cluster

    # --- Step 4: Handle Empty Clusters by Picking the Farthest Data Point ---
    if empty_clusters:
        print(f"Empty clusters detected: {empty_clusters}")
        
        # Compute distances from each user to the closest centroid
        min_distances = np.min([np.linalg.norm(users - centroids[:, c], axis=1) for c in range(centers)], axis=0)

        sorted_indices = np.argsort(min_distances)[::-1]  # Sort users by increasing min distance

        for m in empty_clusters:
            for idx in sorted_indices:
                candidate = tuple(users[idx])
                if candidate not in used_centroids:
                    centroids[:, m] = users[idx]
                    used_centroids.add(candidate)
                    print(f"Reassigned Cluster {m+1} to user index {idx}")
                    break

    # --- Step 5: Check for Convergence ---
    if iteration > 0 and previous_clusters.equals(cluster_assignments):
        print(f"Converged after {iteration + 1} iterations")
        break

    previous_clusters = cluster_assignments.copy()

end = time.time()
print(f"Time taken: {end - start:.2f} seconds")

# Save centroids and assignments
centroids_df = pd.DataFrame(centroids.T, columns=[f"Feature {i+1}" for i in range(users.shape[1])])
centroids_df.to_csv("ProcessedData/centroids.csv", index=False)
cluster_assignments.to_csv("ProcessedData/clusters.csv")

# Plot cluster distribution
cluster_counts = cluster_assignments.value_counts()
plt.figure(figsize=(8, 5))
cluster_counts.sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
plt.xlabel("Cluster")
plt.ylabel("Number of Users")
plt.title("User Distribution Across Clusters")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
