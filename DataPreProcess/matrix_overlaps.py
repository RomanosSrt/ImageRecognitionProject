# compute_overlap_statistics.py

import pandas as pd
import numpy as np
from itertools import combinations
import os
import matplotlib.pyplot as plt

def compute_user_overlap_stats(user_matrix_path="ModifiedData/user_matrix.csv",
                                output_csv="ModifiedData/user_overlap_stats.csv",
                                max_pairs=100000,
                                show_plot=True):
    if not os.path.exists(user_matrix_path):
        print(f"✗ File not found: {user_matrix_path}")
        return

    print("✓ Loading user-movie matrix...")
    ratings_df = pd.read_csv(user_matrix_path, index_col="username")
    binary_matrix = (ratings_df != 0).astype(int)

    print(f"✓ Matrix shape: {binary_matrix.shape} (users × movies)")

    user_list = binary_matrix.index.tolist()
    overlaps = []

    print(f"[*] Computing pairwise overlap percentages (limit {max_pairs} pairs)...")

    for i, (u1, u2) in enumerate(combinations(user_list, 2)):
        if i >= max_pairs:
            break
        vec1 = binary_matrix.loc[u1].values
        vec2 = binary_matrix.loc[u2].values
        intersection = np.sum(vec1 & vec2)
        union = np.sum(vec1 | vec2)
        if union > 0:
            overlaps.append((intersection / union) * 100)

    if not overlaps:
        print("✗ No valid overlaps computed.")
        return

    overlaps = np.array(overlaps)
    print("\n--- Pairwise Overlap Statistics ---")
    print(f"Sample size      : {len(overlaps)} pairs")
    print(f"Mean overlap     : {overlaps.mean():.2f}%")
    print(f"Median overlap   : {np.median(overlaps):.2f}%")
    print(f"10th percentile  : {np.percentile(overlaps, 10):.2f}%")
    print(f"25th percentile  : {np.percentile(overlaps, 25):.2f}%")
    print(f"75th percentile  : {np.percentile(overlaps, 75):.2f}%")
    print(f"Max overlap      : {overlaps.max():.2f}%")

    # Save data
    pd.DataFrame({"overlap_pct": overlaps}).to_csv(output_csv, index=False)
    print(f"\n✓ Saved raw overlap percentages to: {output_csv}")

    # Plot histogram
    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.hist(overlaps, bins=40, color='skyblue', edgecolor='black')
        plt.xlabel("User-User Overlap Percentage")
        plt.ylabel("Number of Pairs")
        plt.title("Distribution of Pairwise User Overlap Percentages")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    compute_user_overlap_stats()
