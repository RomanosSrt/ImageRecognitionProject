from math import sqrt
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def oneHotEncode(data):
    return data.map(lambda x: 1 if x > 0 else 0)

users_file = "ModifiedDataset/user_matrix.csv"  

users = pd.read_csv(users_file)
usernames = users['username']
# print(usernames)
print(users[users['username'] == 'Min111'])
users = users.drop('username', axis=1)

exit()

# λ_users = oneHotEncode(users)
# λ_users.insert(0, 'username', usernames)
# print(λ_users)

centers = 5
centroids = pd.DataFrame(0.0, index=range(centers), columns=users.columns)
centroids = users.sample(n=centers)
centroids = centroids.T
# λ_centroids = oneHotEncode(centroids)
# print(centroids)
# print(λ_centroids)



# kmeans = KMeans(n_clusters=8, max_iter=10)
# kmeans.fit(users)


# cluster_assignments = kmeans.labels_  # Cluster labels for each row
# centroids = kmeans.cluster_centers_  # Centroid of each cluster
# inertia = kmeans.inertia_  # Sum of squared distances to nearest cluster center


# # Output Results
# np.savetxt("ProcessedData/centroids.csv", centroids, delimiter=",", header="Feature1,Feature2,Feature3", comments="")
# np.savetxt("ProcessedData/clusters.csv", cluster_assignments, delimiter=",", comments="")
# print("\nInertia:", inertia)


#---end


distances = pd.DataFrame(0.0, index=usernames.to_list(), columns=[f"Cluster {i+1}" for i in range(centers)])
distance_cosine = pd.DataFrame(0.0, index=usernames.to_list(), columns=[f"Cluster {i+1}" for i in range(centers)])
# distances = np.zeros(len(usernames), centers)

users = users.to_numpy()                    #change to numpy array for faster iterations
centroids = centroids.to_numpy()
# λ_users = λ_users.to_numpy()
# λ_centroids = λ_centroids.to_numpy()


start = time.time()

max_iterations = 10
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
            distances.iloc[j,i] = sqrt(distance) 
            # distance_cosine[j,i] = prod/(sqrt(sum1)*sqrt(sum2)) if sum1 != 0 and sum2 != 0 else 0.0  
    # distances.to_csv(f'ProcessedData/distances{iteration+1}.csv')
    
    if iteration == 0 : distances.to_csv('ProcessedData/distances.csv')
    cluster_assignments = distances.idxmin(axis=1)  # Finds the cluster with the minimum distance for each user
    print(cluster_assignments)
    # if (iteration % 5 == 0):
    # cluster_assignments.to_csv(f'ProcessedData/Clustering{iteration+1}.csv')
    # print(cluster_assignments)
    previous_clusters = cluster_assignments.copy()
    for i in range(centers):
        cluster_users = users[cluster_assignments == f"Cluster {i+1}"]  # Select users in the current cluster
        if len(cluster_users) > 0:
            print(f"Cluster{i+1} : full")
            # print("Before")
            # print(centroids)
            centroids[:, i] = cluster_users.mean(axis=0)
            # print("After")
            # print(centroids)
            if np.all(centroids[:, i] == 0):
                print(f"Warning: Centroid {i+1} is a zero vector")
        else: 
            centroids[:, i] = users[np.random.randint(0, users.shape[0])]
            if np.all(centroids[:, i] == 0):
                print(f"Warning: Centroid {i+1} is a zero vector")
            print(f"Cluster {i+1} : empty")
        
    if (np.array_equal(previous_clusters, cluster_assignments)):  
        print(f"Converged after {iteration + 1} iterations")
        break

end = time.time()
print(f"Time taken: {end - start} seconds")
    

centroids = pd.DataFrame(centroids, columns=[f"Cluster {i+1}" for i in range(centers)])
centroids.to_csv("ProcessedData/centroids.csv")
cluster_assignments.to_csv("ProcessedData/clusters.csv")







# #SOS how many iterations on kmeans? -> https://doi.org/10.1016/j.neucom.2023.126547
# cluster_counts = cluster_assignments.value_counts()

# # Plot bar chart
# plt.figure(figsize=(8, 5))
# cluster_counts.sort_index().plot(kind='bar', color='skyblue', edgecolor='black')

# # Customizing the graph
# plt.xlabel("Cluster")
# plt.ylabel("Number of Users")
# plt.title("User Distribution Across Clusters")
# plt.xticks(rotation=0)  # Keep x-axis labels horizontal
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# # Show the plot
# plt.show()