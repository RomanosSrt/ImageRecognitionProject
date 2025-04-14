import pandas as pd

users_df = pd.read_csv("ModifiedDataset/users.csv")


min_count, max_count = 50, 300  # Set your desired range here
filtered_df = users_df[(users_df['count'] >= min_count) & (users_df['count'] <= max_count)]
filtered_df.reset_index(inplace=True, drop=True)
print(len(filtered_df))
print(filtered_df)



filtered_df.to_csv("ModifiedDataset/users50-300.csv", index=False)
