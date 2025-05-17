# 🎬 Movie Recommender System

This project implements a movie recommendation system based on collaborative filtering techniques, clustering algorithms, and neural networks. It is designed to work with the IMDb Movie Reviews Dataset and was developed as a final project for the course **Pattern Recognition**.

---

## 📦 Overview

The system processes movie review data to:
- Build a user-movie rating matrix
- Cluster users using K-Means and distance metrics (Euclidean, Cosine, Jaccard)
- Identify nearest neighbors within clusters
- Train neural networks to predict user preferences
- Evaluate and visualize model performance

---

## 📁 Project Structure

```

.
├── Dataset/                      # Raw dataset (automatically downloaded if missing)
├── ModifiedData/                # Processed user and movie data, matrix, clusters
├── Prediction/                  # Distance matrices, neighbors, model results
├── setup.py                     # Automatically downloads and extracts the dataset
├── computational\_assignment.py  # Main interactive script with menu
├── user\_matrix.py               # Builds the user-movie rating matrix
├── histogramms.py               # Creates user histograms (ratings and duration)
├── KMeans.py                    # Clusters users using different metrics
├── nearest\_neighbors.py         # Computes top-K neighbors for each user
├── network.py                   # Trains and evaluates neural network models
└── README.md                    # Project documentation

````

---

## 🚀 Getting Started

### 1. Install dependencies:
```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
````

### 2. Run the main script:

```bash
python computational_assignment.py
```

The script will:

* Automatically download and extract the dataset
* Present an interactive menu for running each part of the project

---

## 🧠 Workflow Summary

### Step 1: Build Rating Matrix

* Filters users based on number of reviews
* Builds a matrix of users × movies (`user_matrix.csv`)

### Step 2: Analyze User Behavior

* Generates histograms for review count and rating range

### Step 3: Cluster Users

* Runs K-Means clustering
* Supports Euclidean, Cosine, and Jaccard distances
* Saves `clusters.csv` and visualizations

### Step 4: Find Nearest Neighbors

* Uses intra-cluster distances to find top-K neighbors
* Saves `nearest_neighbors.csv`

### Step 5: Train Neural Networks

* Trains per-cluster networks to predict ratings
* Evaluates model with MAE and MSE
* Saves training loss plots and evaluation metrics

---

## 📊 Key Output Files

| File                             | Description                        |
| -------------------------------- | ---------------------------------- |
| `user_matrix.csv`                | User × movie rating matrix         |
| `clusters.csv`                   | Cluster labels for each user       |
| `distances.csv`                  | User-to-user distance matrix       |
| `nearest_neighbors.csv`          | Top-K neighbors per user           |
| `loss_curve.png`                 | Loss visualization during training |
| `cluster_evaluation_metrics.csv` | MAE/MSE evaluation per cluster     |

---

## 👤 Authors

Developed by:

* **R. Sarantidis** – MPPL2327


---

## 📌 Notes

* The entire pipeline is fully automated and modular
* Dataset is downloaded from Dropbox at runtime
* Each stage of the pipeline is independently callable
* All user inputs (e.g., number of clusters, neighbors) are handled via terminal prompts
