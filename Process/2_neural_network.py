import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Load data
ratings_df = pd.read_csv("ModifiedDataset/user_matrix.csv", index_col="username") / 10.0
neighbors_df = pd.read_csv("Prediction/nearest_neighbors.csv")
clusters_df = pd.read_csv("ProcessedData/clusters.csv")
clusters_df.columns = ["username", "cluster"]
clusters_df.set_index("username", inplace=True)

K = 6
X, y, usernames = [], [], []

# Build feature/label matrix with cluster tracking
for idx, row in neighbors_df.iterrows():
    user = row['username']
    neighbors = row[[f'neighbor_{i+1}' for i in range(K)]]

    if user not in ratings_df.index:
        continue

    user_ratings = ratings_df.loc[user].values
    for movie_idx, target_rating in enumerate(user_ratings):
        if target_rating == 0:
            continue

        neighbor_ratings = []
        valid = True
        for neighbor in neighbors:
            if neighbor in ratings_df.index:
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

X = np.array(X)
y = np.array(y)
usernames = np.array(usernames)

# Train-test split
X_train, X_test, y_train, y_test, users_train, users_test = train_test_split(
    X, y, usernames, test_size=0.2, random_state=42
)

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(K,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=0
)

# Plot training & validation loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Prediction/loss_curve.png")
plt.show()

# Predict
y_pred_train = model.predict(X_train).flatten() * 10
y_true_train = y_train * 10
y_pred_test = model.predict(X_test).flatten() * 10
y_true_test = y_test * 10

# Group users by cluster and compute errors
results = []

for cluster in clusters_df['cluster'].unique():
    cluster_users = clusters_df[clusters_df['cluster'] == cluster].index

    train_mask = np.isin(users_train, cluster_users)
    test_mask = np.isin(users_test, cluster_users)

    if np.sum(train_mask) == 0 or np.sum(test_mask) == 0:
        continue

    mae_train = mean_absolute_error(y_true_train[train_mask], y_pred_train[train_mask])
    mse_train = mean_squared_error(y_true_train[train_mask], y_pred_train[train_mask])
    mae_test = mean_absolute_error(y_true_test[test_mask], y_pred_test[test_mask])
    mse_test = mean_squared_error(y_true_test[test_mask], y_pred_test[test_mask])

    results.append({
        "Cluster": cluster,
        "MAE (Train)": round(mae_train, 4),
        "MSE (Train)": round(mse_train, 4),
        "MAE (Test)": round(mae_test, 4),
        "MSE (Test)": round(mse_test, 4)
    })

# Save results table
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("Prediction/cluster_evaluation_metrics.csv", index=False)
