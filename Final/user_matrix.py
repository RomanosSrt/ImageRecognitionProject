import os
import re
import pandas as pd
import numpy as np

def extract_users_and_movies(folder_path="Dataset/2_reviews_per_movie_raw", output_path="ModifiedData"):
    os.makedirs(output_path, exist_ok=True)
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Extract movie names
    movie_names = [re.sub(r'.csv$', '', f).replace('_', ':') for f in files]
    movie_df = pd.DataFrame(movie_names, columns=['movie_name'])
    movie_df.to_csv(f"{output_path}/movies.csv", index=False)

    # Combine all reviews
    all_reviews = []
    for f in files:
        df = pd.read_csv(os.path.join(folder_path, f))
        df = df[df['rating'] != 'Null']
        all_reviews.append(df)
    combined_df = pd.concat(all_reviews)

    user_counts = combined_df['username'].value_counts().reset_index()
    user_counts.columns = ['username', 'count']
    users_df = pd.DataFrame(sorted(combined_df['username'].dropna().unique()), columns=['username'])
    users_df = pd.merge(users_df, user_counts, on='username', how='left')
    users_df.to_csv(f"{output_path}/users.csv", index=False)

    print(f"✓ Extracted {len(users_df)} users and {len(movie_df)} movies.")

def filter_users_by_range(users_file, min_ratings, max_ratings):
    df = pd.read_csv(users_file)
    filtered = df[(df['count'] >= min_ratings) & (df['count'] <= max_ratings)]
    filtered.to_csv(users_file, index=False)
    print(f"✓ Filtered down to {len(filtered)} users with ratings in range [{min_ratings}, {max_ratings}]")
    return filtered['username'].tolist()

def build_user_movie_matrix(users_file, movies_file, reviews_folder="Dataset/2_reviews_per_movie_raw", output_file="ModifiedData/user_matrix.csv"):
    users = pd.read_csv(users_file)
    movies = pd.read_csv(movies_file)

    user_list = users['username'].tolist()
    movie_list = movies['movie_name'].tolist()

    user_index_map = {user: i for i, user in enumerate(user_list)}
    movie_index_map = {movie: j for j, movie in enumerate(movie_list)}

    matrixR = np.zeros((len(user_list), len(movie_list)), dtype=np.float32)

    for file in os.listdir(reviews_folder):
        movie = re.sub(r'.csv$', '', file).replace('_', ':')
        if movie not in movie_index_map:
            continue
        try:
            ratings = pd.read_csv(os.path.join(reviews_folder, file))
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            continue

        ratings = ratings[ratings['username'].isin(user_index_map)]

        for _, row in ratings.iterrows():
            if pd.notna(row['rating']) and row['rating'] != 'Null':
                matrixR[user_index_map[row['username']], movie_index_map[movie]] = float(row['rating'])

    matrix_df = pd.DataFrame(matrixR, index=user_list, columns=movie_list)
    matrix_df.index.name = 'username'

    # Remove movies with too few ratings
    movie_rating_counts = (matrix_df != 0).sum(axis=0)
    filtered_columns = movie_rating_counts[movie_rating_counts >= 50].index
    matrix_df = matrix_df[filtered_columns]

    matrix_df.to_csv(output_file)
    print(f"✓ Final matrix saved: {matrix_df.shape[0]} users × {matrix_df.shape[1]} movies")

def build_user_matrix():
    print("=== Build User-Movie Matrix ===")
    try:
        min_ratings = int(input("Enter minimum number of ratings per user: "))
        max_ratings = int(input("Enter maximum number of ratings per user: "))
    except ValueError:
        print("Invalid input. Please enter integers.")
        return

    extract_users_and_movies()
    usernames = filter_users_by_range("ModifiedData/users.csv", min_ratings, max_ratings)
    if not usernames:
        print("No users matched the criteria. Aborting.")
        return
    build_user_movie_matrix("ModifiedData/users.csv", "ModifiedData/movies.csv")

if __name__ == "__main__":
    build_user_matrix()
