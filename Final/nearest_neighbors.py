import pandas as pd
import os

def generate_nearest_neighbors(k=6,
                                distances_path="Prediction/distances.csv",
                                clusters_path="ModifiedData/clusters.csv",
                                output_path="Prediction/nearest_neighbors.csv"):
    if not os.path.exists(distances_path):
        print(f"✗ Missing: {distances_path}")
        return
    if not os.path.exists(clusters_path):
        print(f"✗ Missing: {clusters_path}")
        return

    print(f"✓ Loading distances from: {distances_path}")
    distances_df = pd.read_csv(distances_path, index_col=0)
    distances_df.index = distances_df.index.astype(str).str.strip()
    distances_df.columns = distances_df.columns.astype(str).str.strip()

    # Filter to valid usernames in both rows and columns
    valid_usernames = distances_df.index.intersection(distances_df.columns)
    distances_df = distances_df.loc[valid_usernames, valid_usernames]

    clusters_df = pd.read_csv(clusters_path)
    clusters_df.columns = ["username", "cluster"]
    clusters_df["username"] = clusters_df["username"].astype(str).str.strip()
    cluster_series = clusters_df.set_index("username")["cluster"]

    valid_users = distances_df.index.intersection(cluster_series.index)

    neighbors_dict = {
        "username": [],
        "cluster": [],
        **{f"neighbor_{i+1}": [] for i in range(k)}
    }

    for username in valid_users:
        cluster_label = cluster_series[username]
        same_cluster_users = cluster_series[cluster_series == cluster_label].index
        same_cluster_users = same_cluster_users.intersection(distances_df.columns)

        try:
            user_distances = distances_df.loc[username, same_cluster_users]
            user_distances = user_distances.drop(username, errors='ignore')
            user_distances = user_distances[user_distances > 0]  # Exclude self & zero-distance ties
            nearest_neighbors = user_distances.nsmallest(k).index.tolist()

            neighbors_dict["username"].append(username)
            neighbors_dict["cluster"].append(cluster_label)
            for i in range(k):
                neighbors_dict[f"neighbor_{i+1}"].append(
                    nearest_neighbors[i] if i < len(nearest_neighbors) else None
                )

        except KeyError as e:
            print(f"Skipping {username} (missing data): {e}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    neighbors_df = pd.DataFrame(neighbors_dict)
    neighbors_df.to_csv(output_path, index=False)

    print(f"[✓] Nearest neighbors saved to: {output_path}")

if __name__ == "__main__":
    try:
        k = int(input("Enter number of neighbors (default 6): ") or "6")
    except ValueError:
        k = 6
    generate_nearest_neighbors(k=k)
