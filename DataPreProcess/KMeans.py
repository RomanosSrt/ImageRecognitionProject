import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load your dataset
users_file = "ModifiedDataset/user_matrix.csv"
users = pd.read_csv(users_file)

# Drop the username column, assuming it exists
usernames = users['username']
users = users.drop('username', axis=1)

# Initialize KMeans
centers = 4  # number of clusters
kmeans = KMeans(n_clusters=centers, random_state=42)

# Fit the model
kmeans.fit(users)

# Get the cluster centers and the labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Add the labels (cluster assignments) to the original data
users['Cluster'] = labels

# Save the centroids and cluster assignments
centroids_df = pd.DataFrame(centroids, columns=users.columns[:-1])
centroids_df.to_csv("ProcessedData/centroids.csv", index=False)
users.to_csv("ProcessedData/clusters.csv", index=False)

# Plot the distribution of users across clusters
cluster_counts = pd.Series(labels).value_counts()

plt.figure(figsize=(8, 5))
cluster_counts.sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
plt.xlabel("Cluster")
plt.ylabel("Number of Users")
plt.title("User Distribution Across Clusters")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
