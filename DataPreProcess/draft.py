import pandas as pd
import numpy as np
from tqdm import tqdm

# Load binary matrix of ratings (1 = rated, 0 = not rated)
ratings_df = pd.read_csv("ModifiedDataset/user_matrix.csv", index_col=0)
binary_df = (ratings_df > 0).astype(int)

usernames = binary_df.index.tolist()
n = len(usernames)

# Init empty square matrix
jaccard_matrix = pd.DataFrame(np.zeros((n, n)), index=usernames, columns=usernames)

# Compute Jaccard distances
for i in tqdm(range(n), desc="Calculating Jaccard distances"):
    for j in range(i, n):
        u = binary_df.iloc[i].values
        v = binary_df.iloc[j].values

        intersection = np.sum(u & v)
        union = np.sum(u | v)
        dist = 1 - (intersection / union) if union > 0 else 1.0

        jaccard_matrix.iat[i, j] = dist
        jaccard_matrix.iat[j, i] = dist  # symmetric

# Save
jaccard_matrix.to_csv("ProcessedData/distances.csv")
print("[✓] Jaccard distance matrix saved to ProcessedData/jaccard_distances.csv")
