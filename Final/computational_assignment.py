import sys
import os
import pandas as pd
from setup import ensure_dataset_ready
from user_matrix import build_user_matrix
from histogramms import create_histogramms
from KMeans import run_kmeans_clustering
from nearest_neighbors import generate_nearest_neighbors
from network2 import train_network

def main_menu():
    if not os.path.exists("ModifiedData/"):
        os.makedirs("ModifiedData/")
    if not os.path.exists("Prediction/"):
        os.makedirs("Prediction/")
    
    while True:
        print("\n=== Movie Recommender System ===")
        print("1. Build user-movie matrix")
        print("2. Create histograms")
        print("3. Run KMeans clustering")
        print("4. Generate nearest neighbors")
        print("5. Train and evaluate neural network")
        print("6. Exit")

        choice = input("Select an option (1–6): ").strip()

        if choice == "1":
            build_user_matrix()
        elif choice == "2":
            create_histogramms()
        elif choice == "3":
            run_kmeans_clustering()
        elif choice == "4":
            try:
                k = int(input("Enter number of nearest neighbors (default = 6): ") or "6")
            except ValueError:
                k = 6
            generate_nearest_neighbors(k=k)
        elif choice == "5":
            try:
                k = int(input("Enter number of nearest neighbors to use (default = 6): ") or "6")
            except ValueError:
                k = 6

            if not os.path.exists("ModifiedData/clusters.csv"):
                print("⚠ 'clusters.csv' not found. Running KMeans clustering first...")
                run_kmeans_clustering()

            nn_file = "Prediction/nearest_neighbors.csv"
            need_to_generate = True

            if os.path.exists(nn_file):
                neighbors_df = pd.read_csv(nn_file)
                existing_k = sum(col.startswith("neighbor_") for col in neighbors_df.columns)
                if existing_k >= k:
                    need_to_generate = False
                else:
                    print(f"⚠ Found neighbors file but only {existing_k} neighbors. Regenerating...")

            if need_to_generate:
                generate_nearest_neighbors(k=k)

            train_network(K=k)
        elif choice == "6":
            print("Exiting.")
            break
        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    try:
        ensure_dataset_ready()
    except Exception as e:
        print(f"✗ Dataset setup failed: {e}")
        sys.exit(1)

    main_menu()