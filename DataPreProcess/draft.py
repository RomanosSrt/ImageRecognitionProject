import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

users_file = "ModifiedDataset/user_matrix.csv"  
users = pd.read_csv(users_file)
usernames = users.pop('username')  # Drop 'username' column but keep it separately

users = users.to_numpy()  # Convert DataFrame to NumPy array

# Initialize centroids using K-Means++
def kmeans_plus_plus(users, k):
    centroids = [users[np.random.randint(users.shape[0])]]
    for _ in range(1, k):
        dist_sq = np.min([np.sum((users - c) ** 2, axis=1) for c in centroids], axis=0)
        prob = dist_sq / np.sum(dist_sq)
        centroids.append(users[np.random.choice(users.shape[0], p=prob)])
    return np.array(centroids).T

centers = 5
centroids = kmeans_plus_plus(users, centers)

# Initialize distance matrix
distances = np.zeros((users.shape[0], centers))

start = time.time()
max_iterations = 50

for iteration in range(max_iterations):
    print(f"Iteration {iteration+1}")

    # Compute distances using vectorized operations
    for i in range(centers):
        distances[:, i] = np.linalg.norm(users - centroids[:, i].T, axis=1)

    # Assign each user to the closest centroid
    cluster_assignments = np.argmin(distances, axis=1)

    prev_centroids = centroids.copy()
    
    # Update centroids
    for i in range(centers):
        cluster_users = users[cluster_assignments == i]
        if cluster_users.size > 0:
            centroids[:, i] = np.mean(cluster_users, axis=0)

    # Check for convergence (if centroids don’t move significantly)
    if np.linalg.norm(centroids - prev_centroids) < 1e-6:
        print(f"Converged after {iteration+1} iterations")
        break

end = time.time()
print(f"Time taken: {end - start:.4f} seconds")

# Convert results back to Pandas
cluster_assignments = pd.DataFrame(cluster_assignments, index=usernames, columns=["Cluster"])
centroids = pd.DataFrame(centroids, columns=[f"Cluster {i+1}" for i in range(centers)])

# Save results
centroids.to_csv("ProcessedData/centroids.csv")
cluster_assignments.to_csv("ProcessedData/clusters.csv")

# Plot cluster distribution
plt.figure(figsize=(8, 5))
cluster_assignments["Cluster"].value_counts().sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
plt.xlabel("Cluster")
plt.ylabel("Number of Users")
plt.title("User Distribution Across Clusters")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
