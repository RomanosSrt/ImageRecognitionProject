import pandas as pd

data = pd.read_csv('ModifiedDataset/userRatingsMatrix.csv')
user_list = pd.read_csv('ModifiedDataset/userNames5-200.csv')
movies_list = pd.read_csv('ModifiedDataset/movies.csv')

data.columns.values[0] = 'username'
filtered_data = data[data['username'].isin(user_list['username'])]

data = filtered_data

print(data.index)
print(data.columns.tolist())
data = data[sorted(data.columns)]
columns = ['username'] + [col for col in data.columns if col != 'username']
data = data[columns]

data.columns.values[1:] = movies_list['movie_name']

data.to_csv('ModifiedDataset/userRatingsMartix5-200.csv', index=False)