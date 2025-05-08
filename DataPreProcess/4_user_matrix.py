import pandas as pd
import numpy as np
import os
import re

# Paths
users_path = 'ModifiedDataset/users.csv'
movies_path = 'ModifiedDataset/movies.csv'
reviews_folder = 'Dataset/2_reviews_per_movie_raw'

# Load users and movies
users = pd.read_csv(users_path)
movies = pd.read_csv(movies_path)

# Convert usernames and movies to list for indexing
user_list = users['username'].tolist()
movie_list = movies['movie_name'].tolist()

# Create index maps for quick lookup
user_index_map = {user: i for i, user in enumerate(user_list)}
movie_index_map = {movie: j for j, movie in enumerate(movie_list)}

# Initialize ratings matrix with zeros (NumPy array)
matrixR = np.zeros((len(user_list), len(movie_list)), dtype=np.float32)

# Process each review file
for file in os.listdir(reviews_folder):
    # Extract movie name
    movie = re.sub(r'.csv$', '', file).replace('_', ':')

    if movie not in movie_index_map:
        continue  # Skip if movie is not in the list

    # Load the movie's ratings data
    ratings = pd.read_csv(os.path.join(reviews_folder, file))

    # Filter for valid users
    ratings = ratings[ratings['username'].isin(user_index_map)]

    # Convert ratings into NumPy indexing
    for _, row in ratings.iterrows():
        username, rating = row['username'], row['rating']

        
        if pd.notna(rating) and rating != 'Null':
            matrixR[user_index_map[username], movie_index_map[movie]] = float(rating)

# Convert back to Pandas DataFrame if needed
matrixR_df = pd.DataFrame(matrixR, index=user_list, columns=movie_list)
matrixR_df.index.name = 'username'
matrixR_df = matrixR_df.loc[matrixR_df.sum(axis=1) != 0]
print(matrixR_df.head) 
print(len(matrixR))


# Filter movies with at least 50 ratings
movie_rating_counts = (matrixR_df != 0).sum(axis=0)
min_ratings_per_movie = 50
movies_to_keep = movie_rating_counts[movie_rating_counts >= min_ratings_per_movie].index
filtered_matrixR_df = matrixR_df[movies_to_keep]
                           
filtered_matrixR_df.to_csv('ModifiedDataset/user_matrix.csv')
print(filtered_matrixR_df)


#Next Step: Check for all the users that have only null entries as ratings !!SOS!!