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

    # --- NEW: Calculate full user-to-user distance matrix ---
    print("[...] Calculating full user-to-user distance matrix...")
    distance_matrix = pd.DataFrame(index=usernames, columns=usernames, dtype=float)
    for i in range(len(users)):
        for j in range(i, len(users)):
            dist = metric_func(users[i], users[j])
            distance_matrix.iat[i, j] ,distance_matrix.iat[j, i] = dist, dist  # symmetric

    distance_filename = f"Prediction/distances.csv"
    distance_matrix.to_csv(distance_filename)
    print(f"[✓] User distance matrix saved to {distance_filename}")

run_kmeans_clustering(n_clusters=3, distance_metric="jaccard")