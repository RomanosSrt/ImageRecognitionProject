import pandas as pd
import matplotlib.pyplot as plt

# Load per-cluster results
df = pd.read_csv("Prediction/cluster_evaluation_metrics.csv")

# Clean cluster names for axis
df["Cluster"] = df["Cluster"].str.replace("Cluster ", "").astype(int)

# Sort by cluster number
df = df.sort_values("Cluster")

# Plot MAE
plt.figure(figsize=(8, 5))
plt.bar(df["Cluster"], df["MAE (Train)"], width=0.4, label="Train", align="edge")
plt.bar(df["Cluster"] - 0.4, df["MAE (Test)"], width=0.4, label="Test", align="edge")
plt.title("MAE by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Mean Absolute Error")
plt.xticks(df["Cluster"])
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot MSE
plt.figure(figsize=(8, 5))
plt.bar(df["Cluster"], df["MSE (Train)"], width=0.4, label="Train", align="edge")
plt.bar(df["Cluster"] - 0.4, df["MSE (Test)"], width=0.4, label="Test", align="edge")
plt.title("MSE by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Mean Squared Error")
plt.xticks(df["Cluster"])
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
