import os
import numpy as np
import pandas as pd
import re


location = 'Dataset/2_reviews_per_movie_raw'

userRatedMovies = []
userNames = []
concatedData = pd.DataFrame()
for file in os.listdir(location):
    data = pd.read_csv(os.path.join(location, file))
    base_name = re.sub(r' \d{4}\.csv$', '', file)
    base_name = re.sub('_', ':', base_name)
    for i in range(len(data)):
        userRatedMovies.append(base_name)
    
    for name in data['username']:
            userNames.append(name)
    concatedData = pd.concat([concatedData, data], ignore_index=True)
    
concatedData['movie_name'] = userRatedMovies
concatedData['rating'] = concatedData['rating'].apply(lambda rating: rating if rating != 'Null' else 0.0)

user_ids = set(userNames)
item_ids = set(userRatedMovies)

user_map = {user: idx for idx, user in enumerate(user_ids)}
item_map = {item: idx for idx, item in enumerate(item_ids)}

n_users = len(user_ids)
n_items = len(item_ids)
R = pd.DataFrame(0, index=list(user_ids), columns=list(item_ids))

for _, row in concatedData.iterrows():
    user_idx = row['username']
    item_idx = row['movie_name']
    R.at[user_idx, item_idx] = float(row['rating'])

R.to_csv('ModifiedDataset/userRatingsMatrix.csv')
