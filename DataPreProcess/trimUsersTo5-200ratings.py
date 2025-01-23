import os
import pandas as pd

# Paths
folder_path = "Dataset/2_reviews_per_movie_raw"
output_path = "ModifiedDataset"
files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Combine all CSV files into one DataFrame
dataframes = [pd.read_csv(os.path.join(folder_path, file)) for file in files]
combined_df = pd.concat(dataframes, ignore_index=True)

# Count reviews per user
user_counts = combined_df['username'].value_counts().reset_index()
user_counts.columns = ['username', 'count']

# Extract years from the 'date' column for each user
user_dates = combined_df.groupby('username')['date'].apply(
    lambda dates: [pd.to_datetime(date).year for date in dates]
).reset_index()

# Merge counts and dates
result_df = pd.merge(user_counts, user_dates, on='username')

# Filter users by count range
min_count, max_count = 5, 200  # Set your desired range here
filtered_df = result_df[(result_df['count'] >= min_count) & (result_df['count'] <= max_count)]
print(len(filtered_df))

# Save the filtered result
os.makedirs(output_path, exist_ok=True)
filtered_df.to_csv(os.path.join(output_path, "userNames5-200.csv"), index=False)