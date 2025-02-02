import pandas as pd
import os
import re

users = pd.read_csv('ModifiedDataset/users5-200.csv')
movies = pd.read_csv('ModifiedDataset/movies.csv')

matrixR = pd.DataFrame(0.0, index=users['username'], columns=movies['movie_name'])

for file in os.listdir('Dataset/2_reviews_per_movie_raw'):
    movie = re.sub(r' \d{4}\.csv$', '', file)
    movie = re.sub('_', ':', movie)
    ratings = pd.read_csv('Dataset/2_reviews_per_movie_raw/' + file)
    for _, row in ratings.iterrows():
        if row['username'] in matrixR.index and movie in matrixR.columns:
            matrixR.at[row['username'], movie] = (
                float(row['rating']) if row['rating'] != 'Null' else 0.0
            )
        else:
            print(f"Skipped: username={row['username']}, movie={movie}")
                           
matrixR.to_csv('ModifiedDataset/user_matrix.csv')