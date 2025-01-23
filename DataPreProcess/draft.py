import pandas as pd
import random
import os
import re

# Paths
movies_path = 'ModifiedDataset/movies.csv'
users_path = 'ModifiedDataset/users5-200.csv'
user_matrix_path = 'ModifiedDataset/user_matrix.csv'
reviews_folder = 'Dataset/2_reviews_per_movie_raw'

# Load data
movies = pd.read_csv(movies_path)
users = pd.read_csv(users_path)  # Load users.csv
user_matrix = pd.read_csv(user_matrix_path, index_col=0)

# Get the list of valid usernames from users.csv
valid_users = set(users['username'])

# Initialize a report dictionary to store mismatches for each movie
report = {}

# Iterate over all movies
for movie_name in movies['movie_name']:
    # Locate the corresponding CSV file for the movie
    movie_file = None

    # Search for the correct file in the folder
    for file in os.listdir(reviews_folder):
        file_movie_name = re.sub(r' \d{4}\.csv$', '', file)  # Remove " YYYY.csv"
        if file_movie_name == movie_name:
            movie_file = file
            break

    if not movie_file:
        print(f"Movie file for {movie_name} not found in the folder.")
        continue

    # Load the movie's ratings data
    movie_data = pd.read_csv(os.path.join(reviews_folder, movie_file))

    # Filter the movie data to include only users from users.csv
    movie_data = movie_data[movie_data['username'].isin(valid_users)]
    print(f"Loaded {len(movie_data)} reviews for {movie_name} after filtering by users.csv.")

    # Compare ratings for each user in the user matrix
    mismatches = []
    mismatched_users = []  # List to store usernames of mismatched users
    for _, row in movie_data.iterrows():
        username = row['username']
        actual_rating = row['rating']

        # Check if the user exists in the matrix
        if username in user_matrix.index:
            matrix_rating = user_matrix.at[username, movie_name]
            if actual_rating != 'Null' and round(matrix_rating) != int(actual_rating):
                mismatches.append((username, actual_rating, matrix_rating))
                mismatched_users.append(username)  # Add the mismatched username
                print(f"Mismatch Found - User: {username}, Actual: {actual_rating}, Matrix: {matrix_rating}")
        else:
            # Skip usernames not in the user matrix
            continue

    # Ensure mismatched users are part of valid_users only
    mismatched_users = [user for user in mismatched_users if user in valid_users]

    # Add mismatches to the report
    if mismatches:
        report[movie_name] = mismatches

# Step 5: Display Results
if report:
    print("\nMismatch Report:")
    for movie_name, mismatches in report.items():
        print(f"\n{movie_name}:")
        for mismatch in mismatches:
            print(f"User: {mismatch[0]}, Actual: {mismatch[1]}, Matrix: {mismatch[2]}")
else:
    print("All ratings for all movies match the user matrix!")

# Return the report
print("\nReturning the mismatch report.")
print("Mismatch Report:")
print(report)