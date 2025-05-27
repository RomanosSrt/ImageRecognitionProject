import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# === Check and auto-generate matrix ===
def ensure_user_matrix_exists():
    matrix_path = "ModifiedData/user_matrix.csv"
    if not os.path.exists(matrix_path):
        print("⚠ User matrix not found at", matrix_path)
        print("Creating user matrix...")
        try:
            import user_matrix as user_matrix
            user_matrix.build_user_matrix()
        except Exception as e:
            print(f"✗ Failed to build matrix: {e}")
            exit(1)
    else:
        print("✓ Found user matrix.")

# === Distance functions ===
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

def get_distance_function(name):
    return {
        "euclidean": masked_euclidean,
        "cosine": masked_cosine,
        "jaccard": jaccard_distance
    }.get(name)

def kmeans_plus_plus(users, k):
    centroids = [users[np.random.randint(users.shape[0])]]
    for _ in range(1, k):
        dist_sq = np.min([np.sum((users - c) ** 2, axis=1) for c in centroids], axis=0)
        prob = dist_sq / dist_sq.sum()
        centroids.append(users[np.random.choice(users.shape[0], p=prob)])
    return np.array(centroids)

# === Main logic ===
def run_kmeans_clustering():
    ensure_user_matrix_exists()

    df = pd.read_csv("ModifiedData/user_matrix.csv")
    usernames = df['username']
    users = df.drop('username', axis=1).to_numpy()

    try:
        k = int(input("Enter number of clusters (k): "))
    except ValueError:
        print("Invalid input. Please enter an integer.")
        return

    print("Choose a distance metric:")
    print("  1. Euclidean")
    print("  2. Cosine")
    print("  3. Jaccard")
    choice = input("Enter your choice (1-3): ")
    distance_name = {"1": "euclidean", "2": "cosine", "3": "jaccard"}.get(choice)

    if not distance_name:
        print("Invalid choice.")
        return

    metric_func = get_distance_function(distance_name)
    print(f"\n[✓] Running KMeans with k={k}, distance={distance_name}...")

    centroids = kmeans_plus_plus(users, k)
    distances = pd.DataFrame(0.0, index=usernames, columns=[f"Cluster {i+1}" for i in range(k)])
    previous_clusters = None
    max_iterations = 50

    for iteration in range(max_iterations):
        for i in range(k):
            for j in range(users.shape[0]):
                distances.iloc[j, i] = metric_func(users[j], centroids[i])
        cluster_assignments = distances.idxmin(axis=1)

        new_centroids = []
        for m in range(k):
            cluster_users = users[cluster_assignments == f"Cluster {m+1}"]
            if len(cluster_users) > 0:
                mask = cluster_users != 0
                sums = cluster_users.sum(axis=0)
                counts = mask.sum(axis=0)
                avg = np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)
                new_centroids.append(avg)
            else:
                new_centroids.append(users[np.random.randint(users.shape[0])])
        centroids = np.array(new_centroids)

        if previous_clusters is not None and previous_clusters.equals(cluster_assignments):
            print(f"✓ Converged at iteration {iteration}")
            break
        previous_clusters = cluster_assignments.copy()

    # Save results
    os.makedirs("ModifiedData", exist_ok=True)
    pd.DataFrame(centroids).to_csv("ModifiedData/centroids.csv", index=False)
    cluster_assignments.to_csv("ModifiedData/clusters.csv")
    print(f"[✓] Cluster assignments saved to ModifiedData/clusters.csv")

    # Inertia and silhouette
    inertia = sum(
        (metric_func(users[idx], centroids[int(cluster_assignments.iloc[idx].split()[-1]) - 1]) ** 2)
        for idx in range(len(users))
    )
    labels = cluster_assignments.str.extract('(\d+)').astype(int).values.flatten()
    silhouette = silhouette_score(users, labels)

    print(f"Inertia: {inertia:.2f}")
    print(f"Silhouette Score: {silhouette:.4f}")


        # === Create user x user distance matrix ===
    print("[*] Computing user-user distance matrix...")

    n_users = users.shape[0]
    user_dist_matrix = np.zeros((n_users, n_users))

    for i in range(n_users):
        for j in range(i, n_users):  # matrix is symmetric
            dist = metric_func(users[i], users[j])
            user_dist_matrix[i, j] = dist
            user_dist_matrix[j, i] = dist

    dist_df = pd.DataFrame(user_dist_matrix, index=usernames, columns=usernames)
    dist_df.to_csv("Prediction/distances.csv")
    print("[✓] User-user distance matrix saved to Prediction/distances.csv")
    
    # PCA visualization
    pca = PCA(n_components=2)
    users_2d = pca.fit_transform(users)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(users_2d[:, 0], users_2d[:, 1], c=labels, cmap='tab10', s=30)
    plt.title(f"Clustering (k={k}, metric={distance_name})")
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.show()

    # Cluster distribution
    cluster_counts = cluster_assignments.value_counts()
    plt.figure(figsize=(8, 5))
    cluster_counts.sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.xlabel("Cluster")
    plt.ylabel("Number of Users")
    plt.title("User Distribution Across Clusters")
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_kmeans_clustering()
