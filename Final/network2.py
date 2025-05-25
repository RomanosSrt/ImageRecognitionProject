import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K_backend
import os


def custom_accuracy(y_true, y_pred):
    diff = K_backend.abs(y_true - y_pred)
    return K_backend.mean(K_backend.cast(diff <= 0.1, dtype='float32'))


def train_network(K=None):
    if K is None:
        try:
            K = int(input("Enter number of nearest neighbors to use (default = 6): ") or 6)
        except ValueError:
            print("Invalid input. Using default: 6")
            K = 6

    # Load data
    ratings_df = pd.read_csv("ModifiedData/user_matrix.csv", index_col="username") / 10.0
    neighbors_df = pd.read_csv("Prediction/nearest_neighbors.csv")
    clusters_df = pd.read_csv("ModifiedData/clusters.csv")
    clusters_df.columns = ["username", "cluster"]
    clusters_df.set_index("username", inplace=True)

    # Detect max available neighbors
    available_neighbor_cols = [col for col in neighbors_df.columns if col.startswith("neighbor_")]
    max_k_available = len(available_neighbor_cols)
    if K > max_k_available:
        print(f"⚠ Only {max_k_available} neighbors available. Reducing K to {max_k_available}.")
        K = max_k_available

    # Initialize results
    results = []
    os.makedirs("Prediction", exist_ok=True)

    # Train a model for each cluster
    for cluster in clusters_df['cluster'].unique():
        print(f"\n[✓] Training model for {cluster}...")
        cluster_users = clusters_df[clusters_df['cluster'] == cluster].index
        cluster_neighbors_df = neighbors_df[neighbors_df['username'].isin(cluster_users)]

        if cluster_neighbors_df.empty:
            print(f"⚠ No users in {cluster}. Skipping.")
            continue

        X, y, usernames = [], [], []
        # Feature extraction for the cluster
        for idx, row in cluster_neighbors_df.iterrows():
            user = row['username']
            neighbors = row[[f'neighbor_{i+1}' for i in range(K)]]
            neighbors = [n for n in neighbors if pd.notna(n) and str(n).strip() != ""]

            if len(neighbors) < K:
                continue  # Skip this row

            
            if user not in ratings_df.index:
                continue

            user_ratings = ratings_df.loc[user].values
            for movie_idx, target_rating in enumerate(user_ratings):
                if target_rating == 0:
                    continue

                neighbor_ratings = []
                valid = True
                for neighbor in neighbors:
                    if neighbor in ratings_df.index and neighbor in cluster_users:
                        neighbor_rating = ratings_df.loc[neighbor].values[movie_idx]
                        neighbor_ratings.append(neighbor_rating)
                    else:
                        valid = False
                        break

                if not valid or np.all(np.array(neighbor_ratings) == 0):
                    continue

                X.append(neighbor_ratings)
                y.append(target_rating)
                usernames.append(user)

        if not X:
            print(f"⚠ No valid data for {cluster}. Skipping.")
            continue

        X = np.array(X)
        y = np.array(y)
        usernames = np.array(usernames)

        # Train-test split
        X_train, X_test, y_train, y_test, users_train, users_test = train_test_split(
            X, y, usernames, test_size=0.2, random_state=42
        )

        # Build neural network
        model = Sequential([
            Input(shape=(K,)),
            # Dense(256, activation='relu'),
            # Dropout(0.45),
            # Dense(128, activation='relu'),
            # Dropout(0.45),
            Dense(64, activation='relu'),
            # Dropout(0.45),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=[custom_accuracy])

        # Train model
        early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0
        )

        # Plot loss curve
        plt.figure(figsize=(8, 5))
        plt.plot(history.history['loss'], label="Training Loss")
        plt.plot(history.history['val_loss'], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.title(f"Training vs Validation Loss ({cluster})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"Prediction/loss_curve_{cluster}.png")
        plt.close()
        
        # Plot accuracy
        plt.figure(figsize=(8, 5))
        plt.plot(history.history['custom_accuracy'], label="Training Accuracy")
        plt.plot(history.history['val_custom_accuracy'], label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (±0.5 rating)")
        plt.title(f"Training vs Validation Accuracy ({cluster})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"Prediction/accuracy_curve_{cluster}.png")
        plt.close()


        # Evaluate model
        y_pred_train = model.predict(X_train).flatten() * 10
        y_true_train = y_train * 10
        y_pred_test = model.predict(X_test).flatten() * 10
        y_true_test = y_test * 10

        mae_train = mean_absolute_error(y_true_train, y_pred_train)
        mse_train = mean_squared_error(y_true_train, y_pred_train)
        train_accuracy = np.mean(np.abs(y_pred_train - y_true_train) <= 1)
        mae_test = mean_absolute_error(y_true_test, y_pred_test)
        mse_test = mean_squared_error(y_true_test, y_pred_test)
        test_accuracy = np.mean(np.abs(y_pred_test - y_true_test) <= 1)


        results.append({
            "Cluster": cluster,
            "MAE (Train)": round(mae_train, 4),
            "MSE (Train)": round(mse_train, 4),
            "Accuracy (Train)": round(train_accuracy, 4),
            "MAE (Test)": round(mae_test, 4),
            "MSE (Test)": round(mse_test, 4),
            "Accuracy (Test)": round(test_accuracy, 4)
        })

    # Save results
    results_df = pd.DataFrame(results)
    print("\nEvaluation Metrics per Cluster:")
    print(results_df)
    results_df.to_csv("Prediction/cluster_evaluation_metrics.csv", index=False)

if __name__ == "__main__":
    train_network()