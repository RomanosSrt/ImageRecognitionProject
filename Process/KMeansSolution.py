def run_kmeans_clustering(n_clusters=3, distance_metric="euclidean"):
    import numpy as np
    import pandas as pd
    import time
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA

    # Load data
    users_df = pd.read_csv("ModifiedDataset/user_matrix.csv")
    usernames = users_df['username']
    users = users_df.drop('username', axis=1).to_numpy()

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

    metric_func = {
        "euclidean": masked_euclidean,
        "cosine": masked_cosine,
        "jaccard": jaccard_distance
    }[distance_metric]

    # K-means++ initialization
    def kmeans_plus_plus(users, k):
        centroids = [users[np.random.randint(users.shape[0])]]
        for _ in range(1, k):
            dist_sq = np.min([np.sum((users - c) ** 2, axis=1) for c in centroids], axis=0)
            prob = dist_sq / dist_sq.sum()
            centroids.append(users[np.random.choice(users.shape[0], p=prob)])
        return np.array(centroids)

    centroids = kmeans_plus_plus(users, n_clusters)
    distances = pd.DataFrame(0.0, index=usernames, columns=[f"Cluster {i+1}" for i in range(n_clusters)])
    previous_clusters = None
    max_iterations = 50

    for iteration in range(max_iterations):
        for i in range(n_clusters):
            for j in range(users.shape[0]):
                distances.iloc[j, i] = metric_func(users[j], centroids[i])
        cluster_assignments = distances.idxmin(axis=1)

        new_centroids = []
        for m in range(n_clusters):
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
            print(f"Converged at iteration {iteration}")
            break
        previous_clusters = cluster_assignments.copy()

    # Save results
    pd.DataFrame(centroids).to_csv("ProcessedData/centroids.csv", index=False)
    cluster_assignments.to_csv("ProcessedData/clusters.csv")

    # Inertia
    inertia = sum(
        (metric_func(users[idx], centroids[int(cluster_assignments.iloc[idx].split()[-1]) - 1]) ** 2)
        for idx in range(len(users))
    )
    print(f"Inertia: {inertia:.2f}")

    # Silhouette
    labels = cluster_assignments.str.extract('(\d+)').astype(int).values.flatten()
    silhouette = silhouette_score(users, labels)
    print(f"Silhouette Score: {silhouette:.4f}")

    # Plot clusters (PCA projection)
    pca = PCA(n_components=2)
    users_2d = pca.fit_transform(users)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(users_2d[:, 0], users_2d[:, 1], c=labels, cmap='tab10', s=30)
    plt.title(f"Clustering (k={n_clusters}, metric={distance_metric})")
    plt.colorbar(scatter)
    plt.show()

    # Plot cluster counts
    cluster_counts = cluster_assignments.value_counts()
    plt.figure(figsize=(8, 5))
    cluster_counts.sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.xlabel("Cluster")
    plt.ylabel("Number of Users")
    plt.title("User Distribution Across Clusters")
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # --- NEW: Calculate full user-to-user distance matrix ---
    print("[...] Calculating full user-to-user distance matrix...")
    distance_matrix = pd.DataFrame(index=usernames, columns=usernames, dtype=float)
    for i in range(len(users)):
        for j in range(i, len(users)):
            dist = metric_func(users[i], users[j])
            distance_matrix.iat[i, j] = dist
            distance_matrix.iat[j, i] = dist  # symmetric

    distance_filename = f"ProcessedData/distances.csv"
    distance_matrix.to_csv(distance_filename)
    print(f"[✓] User distance matrix saved to {distance_filename}")

    return inertia, silhouette, cluster_assignments
