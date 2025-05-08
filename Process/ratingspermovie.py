import pandas as pd
import matplotlib.pyplot as plt

# Load filtered users
users_df = pd.read_csv("ModifiedDataset/users50-300.csv")
filtered_usernames = set(users_df['username'])

# Load user-movie ratings matrix
ratings_df = pd.read_csv("ModifiedDataset/user_matrix.csv")

# Filter only the selected users (50-300 ratings)
ratings_df = ratings_df[ratings_df['username'].isin(filtered_usernames)]
ratings_df = ratings_df.set_index('username')

# Count non-zero ratings per movie (column-wise)
movie_rating_counts = (ratings_df != 0).sum(axis=0)

# Sort movie counts descending
movie_rating_counts = movie_rating_counts.sort_values(ascending=False)

# Plot
plt.figure(figsize=(12, 6))
plt.bar(range(len(movie_rating_counts)), movie_rating_counts.values, edgecolor='black')
plt.xlabel('Movies (sorted)')
plt.ylabel('Number of Ratings')
plt.title('Number of Ratings per Movie (Users with 50-300 ratings)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
