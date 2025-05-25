import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from KMeans import (
    ensure_user_matrix_exists,
    get_distance_function,
    kmeans_plus_plus,
    run_kmeans_clustering
)

def run_kmeans(users, usernames, k, distance_name):
    metric_func = get_distance_function(distance_name)
    centroids = kmeans_plus_plus(users, k)
    distances = pd.DataFrame(0.0, index=usernames, columns=[f"Cluster {i+1}" for i in range(k)])
    previous_clusters = None

    for _ in range(50):
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
            break
        previous_clusters = cluster_assignments.copy()

    labels = cluster_assignments.str.extract('(\d+)').astype(int).values.flatten()
    inertia = sum(
        (metric_func(users[idx], centroids[int(cluster_assignments.iloc[idx].split()[-1]) - 1]) ** 2)
        for idx in range(len(users))
    )
    silhouette = silhouette_score(users, labels) if k > 1 else None

    return inertia, silhouette

def main():
    ensure_user_matrix_exists()
    df = pd.read_csv("ModifiedData/user_matrix.csv")
    usernames = df["username"]
    users = df.drop("username", axis=1).to_numpy()

    distance_metric = "jaccard"  # Change to "cosine" or "jaccard" as needed
    k_values = list(range(2, 11))  # Test k from 2 to 10
    inertias = []
    silhouettes = []

    print(f"🔍 Testing optimal k for distance metric: {distance_metric}")

    for k in k_values:
        print(f"  → Running k={k}")
        inertia, silhouette = run_kmeans(users, usernames, k, distance_metric)
        inertias.append(inertia)
        silhouettes.append(silhouette)
        print(f"    Inertia: {inertia:.2f} | Silhouette: {silhouette:.4f}")

    # Plotting results
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(k_values, inertias, marker='o', color='blue')
    plt.title("Inertia vs k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")

    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouettes, marker='o', color='green')
    plt.title("Silhouette Score vs k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")

    plt.tight_layout()
    plt.savefig("ModifiedData/optimal_k_plot.png")
    plt.show()

if __name__ == "__main__":
    # main()
    run_kmeans_clustering()
