import os
import re
import pandas as pd

def extract_and_filter_users():
    folder_path = "Dataset/2_reviews_per_movie_raw"
    output_path = "ModifiedDataset"
    os.makedirs(output_path, exist_ok=True)

    print("=== Extracting unique users and movies ===")

    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Extract movie names
    movieNames = []
    for file in files:
        movieName = re.sub(r'.csv$', '', file)
        movieName = re.sub('_', ':', movieName)
        movieNames.append(movieName)

    movieNames_df = pd.DataFrame(movieNames, columns=['movie_name'])
    movieNames_df.to_csv(os.path.join(output_path, 'movies.csv'), index=False)

    # Combine review data
    dataframes = []
    for file in files:
        df = pd.read_csv(os.path.join(folder_path, file))
        df = df[df['rating'] != 'Null']
        dataframes.append(df)

    combined_df = pd.concat(dataframes)
    user_counts = combined_df['username'].value_counts().reset_index()
    user_counts.columns = ['username', 'count']

    # Build user list with counts
    unique_users_df = pd.DataFrame(sorted(combined_df['username'].dropna().unique()), columns=['username'])
    users_df = pd.merge(unique_users_df, user_counts, on='username', how='left')

    print(f"\nTotal unique users before filtering: {len(users_df)}")

    # Ask for filtering range
    try:
        min_count = int(input("Enter minimum number of ratings per user (e.g. 50): "))
        max_count = int(input("Enter maximum number of ratings per user (e.g. 400): "))
    except ValueError:
        print("Invalid input. Using default range 50–400.")
        min_count, max_count = 50, 400

    filtered_df = users_df[(users_df['count'] >= min_count) & (users_df['count'] <= max_count)]
    filtered_df.reset_index(inplace=True, drop=True)

    print(f"Users retained after filtering: {len(filtered_df)}")
    print(filtered_df.head())

    # Save results
    filtered_df.to_csv(os.path.join(output_path, 'users.csv'), index=False)

if __name__ == "__main__":
    extract_and_filter_users()
