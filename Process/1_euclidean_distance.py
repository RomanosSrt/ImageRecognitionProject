from math import sqrt
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans



users_file = "ModifiedDataset/user_matrix.csv"  

users = pd.read_csv(users_file)
print(users.head)
# exit()
usernames = users['username']

users = users.drop('username', axis=1)


# λ_users.insert(0, 'username', usernames)
# print(λ_users)

centers = 4
centroids = pd.DataFrame(0.0, index=range(centers), columns=users.columns)
centroids = users.sample(n=centers)
centroids = centroids.T
# print(centroids)
# print(λ_centroids)


# # Output Results
# np.savetxt("ProcessedData/centroids.csv", centroids, delimiter=",", header="Feature1,Feature2,Feature3", comments="")
# np.savetxt("ProcessedData/clusters.csv", cluster_assignments, delimiter=",", comments="")
# print("\nInertia:", inertia)


#---end


distances = pd.DataFrame(0.0, index=usernames.to_list(), columns=[f"Cluster {i+1}" for i in range(centers)])
distance_cosine = pd.DataFrame(0.0, index=usernames.to_list(), columns=[f"Cluster {i+1}" for i in range(centers)])
# distances = np.zeros(len(usernames), centers)

users = users.to_numpy()                    #change to numpy array for faster iterations


def kmeans_plus_plus(users, k):
    """Initialize centroids using the K-Means++ method."""
    centroids = []
    centroids.append(users[np.random.randint(users.shape[0])])  # Pick first randomly

    for _ in range(1, k):
        # Compute squared distances from existing centroids
        dist_sq = np.min([np.sum((users - c) ** 2, axis=1) for c in centroids], axis=0)
        prob = dist_sq / dist_sq.sum()  # Probability distribution
        chosen_idx = np.random.choice(users.shape[0], p=prob)
        centroids.append(users[chosen_idx])

    return np.array(centroids).T  # Transpose to match expected shape

centroids = kmeans_plus_plus(users, centers)

# centroids = centroids.to_numpy()
# λ_users = λ_users.to_numpy()
# λ_centroids = λ_centroids.to_numpy()



start = time.time()

max_iterations = 50
for iteration in range(max_iterations):
    print(f"{iteration+1} - Clustering cycle")
    for i in range(centers):
        # print(f"Calculating distances for Cluster {i+1}")
        for j in range(len(usernames)):
            distance = 0
            # prod = 0
            # sum1 = 0
            # sum2 = 0
            for k in range(len(centroids)):
                if users[j,k] != 0 and centroids[k,i] != 0:               #instead of iterating through the λ tables less time
                    distance += (users[j,k] - centroids[k,i]) ** 2
                    # prod += users[j,k]*centroids[k,i]
                    # sum1 += users[j,k]**2
                    # sum2 += centroids[k,i]**2

                    
            distances.iloc[j,i] = np.sqrt(distance) if distance != 0 else np.inf
            # distance_cosine[j,i] = prod/(sqrt(sum1)*sqrt(sum2)) if sum1 != 0 and sum2 != 0 else 0.0  
    # distances.to_csv(f'ProcessedData/distances{iteration+1}.csv')

    
    if iteration == 0 : distances.to_csv('ProcessedData/distances.csv')

    cluster_assignments = distances.idxmin(axis=1)  # Finds the cluster with the minimum distance for each user
    # if (iteration % 5 == 0):
    # cluster_assignments.to_csv(f'ProcessedData/Clustering{iteration+1}.csv')
    # print(cluster_assignments)
    used_centroids = set()
    empty_clusters = []
    for m in range(centers):
        cluster_users = users[cluster_assignments == f"Cluster {m + 1}"]  # Select users in the current cluster
        if len(cluster_users) > 0:
            print(f"Cluster{m + 1} : ", len(cluster_users))
            # print("Before")
            # print(centroids)
            centroids[:, m] = cluster_users.mean(axis=0)
            # print("After")
            # print(centroids)
            if np.all(centroids[:, m] == 0):
                print(f"Warning: Centroid {m + 1} is a zero vector")
        else: 
            empty_clusters.append(m)

    if empty_clusters:
        print(f"Empty clusters detected: {empty_clusters}")
        
        # Compute distances from each user to the closest centroid
        min_distances = np.min([np.linalg.norm(users - centroids[:, c], axis=1) for c in range(centers)], axis=0)

        sorted_indices = np.argsort(min_distances)[::-1]  # Sort users by increasing min distance

        for m in empty_clusters:
            for idx in sorted_indices:
                candidate = tuple(users[idx])
                if candidate not in used_centroids:
                    centroids[:, m] = users[idx]
                    used_centroids.add(candidate)
                    print(f"Reassigned Cluster {m+1} to user index {idx}")
                    break
            
            
            
            
            
            
            # norms = np.linalg.norm(users, axis=1)       #euclidean distance of all vectors

            # longest_vector_index = np.argmax(norms)            
            # centroids[:, m] = users[longest_vector_index]
            # # cluster_assignments[]
             
            # centroids[:, m] = users[np.random.randint(0, users.shape[0])]
            # if np.all(centroids[:, m] == 0):
            #     print(f"Warning: Centroid {m+1} is a zero vector")
            # print(f"Cluster {m+1} : empty")
    
    if iteration > 0 and np.array_equal(previous_clusters.values, cluster_assignments.values):  
        print(f"Converged after {iteration + 1} iterations")
        break
    previous_clusters = cluster_assignments.copy()

end = time.time()
print(f"Time taken: {end - start} seconds")
    

centroids = pd.DataFrame(centroids, columns=[f"Cluster {i+1}" for i in range(centers)])
centroids.to_csv("ProcessedData/centroids.csv")
cluster_assignments.to_csv("ProcessedData/clusters.csv")

from sklearn.metrics import silhouette_score

centroids_np = centroids.to_numpy()  # Ensure centroids are NumPy array

# Calculate Inertia
inertia = 0
for idx, cluster_label in enumerate(cluster_assignments):
    cluster_idx = int(cluster_label.split()[-1]) - 1  # Convert "Cluster X" to index
    user_vector = users[idx]
    centroid_vector = centroids_np[:, cluster_idx]
    distance = np.sum((user_vector - centroid_vector) ** 2)
    inertia += distance

print(f"Inertia: {inertia}")

# Calculate Silhouette Score
labels = cluster_assignments.str.extract('(\d+)').astype(int).values.flatten()
silhouette_avg = silhouette_score(users, labels)
print(f"Silhouette Score: {silhouette_avg}")







# #SOS how many iterations on kmeans? -> https://doi.org/10.1016/j.neucom.2023.126547
cluster_counts = cluster_assignments.value_counts()

# Plot bar chart
plt.figure(figsize=(8, 5))
cluster_counts.sort_index().plot(kind='bar', color='skyblue', edgecolor='black')

# Customizing the graph
plt.xlabel("Cluster")
plt.ylabel("Number of Users")
plt.title("User Distribution Across Clusters")
plt.xticks(rotation=0)  # Keep x-axis labels horizontal
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()