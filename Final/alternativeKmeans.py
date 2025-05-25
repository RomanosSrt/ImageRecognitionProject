import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import umap
from sklearn.cluster import KMeans, SpectralClustering
import hdbscan

def ensure_user_matrix_exists():
    matrix_path = "ModifiedData/user_matrix.csv"
    if not os.path.exists(matrix_path):
        print("⚠ User matrix not found at", matrix_path)
        print("Creating user matrix...")
        try:
            import Final.user_matrix as user_matrix
            user_matrix.main()
        except Exception as e:
            print(f"✗ Failed to build matrix: {e}")
            exit(1)
    else:
        print("✓ Found user matrix.")

def run_best_clustering():
    ensure_user_matrix_exists()

    df = pd.read_csv("ModifiedData/user_matrix.csv")
    usernames = df['username']
    users = df.drop('username', axis=1).to_numpy()

    print(f"Loaded user matrix with {users.shape[0]} users and {users.shape[1]} features")

    print("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_components=100, random_state=42)
    umap_data = reducer.fit_transform(users)

    clustering_results = []

    # 1) HDBSCAN (min_cluster_size=10 as example, expect ~16 clusters)
    print("Testing HDBSCAN...")
    try:
        hdb = hdbscan.HDBSCAN(min_cluster_size=50, metric="euclidean")
        hdb_labels = hdb.fit_predict(umap_data)
        filtered_mask = hdb_labels != -1
        if np.sum(filtered_mask) > 0 and len(np.unique(hdb_labels[filtered_mask])) > 1:
            score = silhouette_score(umap_data[filtered_mask], hdb_labels[filtered_mask])
            clustering_results.append(("HDBSCAN", hdb_labels, score))
            print(f"HDBSCAN silhouette: {score:.4f}")
        else:
            print("HDBSCAN produced mostly noise or a single cluster.")
    except Exception as e:
        print(f"HDBSCAN error: {e}")

    # 2) KMeans k=2
    print("Testing KMeans (k=2)...")
    try:
        kmeans = KMeans(n_clusters=4, random_state=42)
        km_labels = kmeans.fit_predict(umap_data)
        score = silhouette_score(umap_data, km_labels)
        clustering_results.append(("KMeans", km_labels, score))
        print(f"KMeans silhouette: {score:.4f}")
    except Exception as e:
        print(f"KMeans error: {e}")

    # 3) Spectral Clustering k=2
    print("Testing Spectral Clustering (k=2)...")
    try:
        spectral = SpectralClustering(n_clusters=5, affinity="nearest_neighbors", assign_labels="kmeans", random_state=42)
        sp_labels = spectral.fit_predict(umap_data)
        score = silhouette_score(umap_data, sp_labels)
        clustering_results.append(("Spectral", sp_labels, score))
        print(f"Spectral silhouette: {score:.4f}")
    except Exception as e:
        print(f"Spectral error: {e}")

    if not clustering_results:
        print("No clustering results obtained. Exiting.")
        return

    best_algo, best_labels, best_score = max(clustering_results, key=lambda x: x[2])
    print(f"\nBest clustering algorithm: {best_algo} with silhouette score {best_score:.4f}")

    # Save clusters.csv with columns: username, cluster
    # ... after choosing best_algo, best_labels, best_score ...

    unique_labels = np.unique(best_labels)
    label_map = {label: f"Cluster {i+1}" for i, label in enumerate(sorted(unique_labels))}
    cluster_str_labels = [label_map[label] for label in best_labels]

    cluster_df = pd.DataFrame({
        "username": usernames,
        "cluster": cluster_str_labels
    })

    os.makedirs("ModifiedData", exist_ok=True)
    cluster_df.to_csv("ModifiedData/clusters.csv", index=False)
    print("Saved ModifiedData/clusters.csv")

        # === Compute user-user Euclidean distance matrix on UMAP data ===
    print("[*] Computing user-user distance matrix on UMAP-reduced data...")

    from scipy.spatial.distance import cdist
    dist_matrix = cdist(umap_data, umap_data, metric='euclidean')
    dist_df = pd.DataFrame(dist_matrix, index=usernames, columns=usernames)
    dist_df.to_csv("Prediction/distances.csv")
    print("[✓] User-user distance matrix saved to Prediction/distances.csv")


    # Optional: PCA plot of clustering
    pca = PCA(n_components=2)
    users_2d = pca.fit_transform(users)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(users_2d[:,0], users_2d[:,1], c=best_labels, cmap='tab10', s=30)
    plt.title(f"Clustering by {best_algo} (Silhouette={best_score:.4f})")
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_best_clustering()
