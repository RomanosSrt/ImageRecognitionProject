from math import sqrt
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def oneHotEncode(data):
    return data.map(lambda x: 1 if x > 0 else 0)

users_file = "ModifiedDataset/user_matrix.csv"  

users = pd.read_csv(users_file)
usernames = users['username']
users = users.drop('username', axis=1)

centers = 5
centroids = pd.DataFrame(0.0, index=range(centers), columns=users.columns)
centroids = users.sample(n=centers)
centroids = centroids.T


distances = pd.DataFrame(0.0, index=usernames.to_list(), columns=[f"Cluster {i+1}" for i in range(centers)])
distance_cosine = pd.DataFrame(0.0, index=usernames.to_list(), columns=[f"Cluster {i+1}" for i in range(centers)])

users = users.to_numpy()                    #change to numpy array for faster iterations
centroids = centroids.to_numpy()


start = time.time()

max_iterations = 10
for iteration in range(max_iterations):
    print(f"{iteration+1} - Clustering cycle")
    for i in range(centers):
        for j in range(len(usernames)):
            prod = 0
            sum1 = 0
            sum2 = 0
            for k in range(len(centroids)):
                if users[j,k] != 0 and centroids[k,i] != 0:               #instead of iterating through the λ tables less time
                    prod += users[j,k]*centroids[k,i]
                    sum1 += users[j,k]**2
                    sum2 += centroids[k,i]**2
            distance_cosine[j,i] = prod/(sqrt(sum1)*sqrt(sum2)) if sum1 != 0 and sum2 != 0 else 0.0  
    cluster_assignments = distance_cosine.idxmin(axis=1)  # Finds the cluster with the minimum distance for each user

    previous_clusters = cluster_assignments.copy()
    for i in range(centers):
        cluster_users = users[cluster_assignments == f"Cluster {i+1}"]  # Select users in the current cluster
        if len(cluster_users) > 0:
            print(f"Cluster {i+1} : full")
            centroids[:, i] = cluster_users.mean(axis=0)
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
