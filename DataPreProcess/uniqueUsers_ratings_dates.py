import os
import pandas as pd

folder_path = "Dataset/2_reviews_per_movie_raw"
output_path = "ModifiedDataset"
files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

dataframes = []
for file in files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    dataframes.append(df)


combined_df = pd.concat(dataframes)

if 'username' in combined_df.columns and 'date' in combined_df.columns:
    user_counts = combined_df['username'].value_counts()
    print("Υπάρχει 𝛷:𝐔 → 𝑷(𝜤) τέτοια ώστε ∀ 𝑢 ∈ 𝑼̂ ισχύει ότι: " + str(user_counts.min()) + " ≤ |𝜑(𝑢)| ≤ " + str(user_counts.max()))

    user_dates = combined_df.groupby('username')['date'].apply(list).reset_index()
    user_dates['date'] = user_dates['date'].apply(lambda dates: [pd.to_datetime(date).year for date in dates])
    user_counts_df = user_counts.reset_index()
    user_counts_df.columns = ['username', 'count']
    result_df = pd.merge(user_counts_df, user_dates, on='username')

    result_df.to_csv(os.path.join(output_path, "userNames.csv"), index=False)
else:
    print("'username' or 'date' column not found in combined DataFrame")


print(combined_df.head)