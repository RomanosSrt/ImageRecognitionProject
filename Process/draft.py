from KMeansSolution import run_kmeans_clustering
import pandas as pd

users_df = pd.read_csv("ModifiedDataset/user_matrix.csv")

metrics = ["euclidean", "cosine", "jaccard"]

run_kmeans_clustering(n_clusters=3, distance_metric="jaccard")
exit(0)

for metric in metrics:
    print(f"Running KMeans with {metric} distance metric")
    print("k=3")
    run_kmeans_clustering(n_clusters=11, distance_metric=metric)
    exit(0)
    print("k=4")
    run_kmeans_clustering(n_clusters=4, distance_metric=metric)
    print("k=5")
    run_kmeans_clustering(n_clusters=5, distance_metric=metric)
    print("===========================")