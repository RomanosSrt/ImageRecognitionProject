import os
import re
import pandas as pd

folder_path = "Dataset/2_reviews_per_movie_raw"
output_path = "../ModifiedDataset"
files = [f for f in os.listdir(folder_path)]

movieNames = []
for file in files:
    movieName = re.sub(r'\s?\d{4}.csv$', '', file)
    movieName = re.sub('_', ':', movieName)
    movieNames.append(movieName)
    

        
movieNames_df = pd.DataFrame(movieNames, columns=['movie_name'])
movieNames_df.to_csv('ModifiedDataset/movies.csv', index=False)