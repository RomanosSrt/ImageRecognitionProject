import pandas as pd

# Παράμετροι
k = 6
distances_path = "Prediction/distances.csv"
clusters_path = "ProcessedData/clusters.csv"
output_path = "Prediction/nearest_neighbors.csv"

# Φόρτωση και καθαρισμός πίνακα αποστάσεων
distances_df = pd.read_csv(distances_path, index_col=0)
distances_df.index = distances_df.index.astype(str).str.strip()
distances_df.columns = distances_df.columns.astype(str).str.strip()

# Φιλτράρισμα: κράτα μόνο στήλες που είναι usernames (που υπάρχουν και στο index)
valid_usernames = distances_df.index.intersection(distances_df.columns)
distances_df = distances_df.loc[valid_usernames, valid_usernames]

# Φόρτωση και καθαρισμός πίνακα συστάδων
clusters_df = pd.read_csv(clusters_path)
clusters_df.columns = ["username", "cluster"]
clusters_df["username"] = clusters_df["username"].astype(str).str.strip()

# Σειριοποίηση για ταχύτερη πρόσβαση
cluster_series = clusters_df.set_index("username")["cluster"]

# Τελική λίστα valid χρηστών
valid_users = distances_df.index.intersection(cluster_series.index)

# Λεξικό εξόδου
neighbors_dict = {
    "username": [],
    "cluster": [],
    **{f"neighbor_{i+1}": [] for i in range(k)}
}

# Εύρεση γειτόνων
for username in valid_users:
    cluster_label = cluster_series[username]
    same_cluster_users = cluster_series[cluster_series == cluster_label].index
    same_cluster_users = same_cluster_users.intersection(distances_df.columns)

    # Αποστάσεις προς άλλους χρήστες του ίδιου cluster
    try:
        user_distances = distances_df.loc[username, same_cluster_users]
        user_distances = user_distances.drop(username, errors='ignore')
        nearest_neighbors = user_distances.nsmallest(k).index.tolist()

        neighbors_dict["username"].append(username)
        neighbors_dict["cluster"].append(cluster_label)
        for i in range(k):
            neighbors_dict[f"neighbor_{i+1}"].append(nearest_neighbors[i] if i < len(nearest_neighbors) else None)

    except KeyError as e:
        print(f"Skipping {username} (missing data): {e}")

# Αποθήκευση
neighbors_df = pd.DataFrame(neighbors_dict)
neighbors_df.to_csv(output_path, index=False)
print(f"[✓] Nearest neighbors saved to: {output_path}")
print("Example index values:")
print(distances_df.index[:5].tolist())

print("\nExample column values:")
print(distances_df.columns[:5].tolist())

print("\nColumns not found in index:")
print(set(distances_df.columns) - set(distances_df.index))
