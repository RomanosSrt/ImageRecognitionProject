import os
import re
import pandas as pd

folder_path = "Dataset/2_reviews_per_movie_raw"
output_path = "../ModifiedDataset"
os.makedirs(output_path, exist_ok=True)
files = [f for f in os.listdir(folder_path)]

movieNames = []
for file in files:
    movieName = re.sub(r'\s?\d{4}.csv$', '', file)   # Remove the year.csv from the file name
    movieName = re.sub('_', ':', movieName)          # Replace underscores with colons
    movieNames.append(movieName)

movieNames_df = pd.DataFrame(movieNames, columns=['movie_name'])
movieNames_df.to_csv('ModifiedDataset/movies.csv', index=False)

folder_path = "Dataset/2_reviews_per_movie_raw"
files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

dataframes = []
for file in files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    df = df[df['rating'] != 'Null']                 #remove all entries of null
    dataframes.append(df)
    
combined_df = pd.concat(dataframes)
user_counts = combined_df['username'].value_counts().reset_index()
user_counts.columns = ['username', 'count']

combined_df = pd.DataFrame(sorted(combined_df['username'].dropna().unique()), columns=['username'])
combined_df = pd.merge(combined_df, user_counts, on='username', how='left')
combined_df.to_csv('ModifiedDataset/users.csv', index=False)