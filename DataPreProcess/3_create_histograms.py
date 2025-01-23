import pandas as pd
import os
import ast

users_file = "ModifiedDataset/users5-200.csv"  # Path to your users.csv
movies_folder = "Dataset/2_reviews_per_movie_raw"  # Folder containing movie CSV files
output_file = "ModifiedDataset/user_dates5-200.csv"  # Where to save the result

# # Read the users.csv file
# users_df = pd.read_csv(users_file)
# usernames = set(users_df['username'])  # Get the set of usernames to filter by

# # Initialize a dictionary to store dates for each user
# user_dates = {username: [] for username in usernames}

# # Iterate over all movie files
# for movie_file in os.listdir(movies_folder):
#     if movie_file.endswith('.csv'):  # Process only CSV files
#         movie_path = os.path.join(movies_folder, movie_file)
#         movie_df = pd.read_csv(movie_path)

#         # Filter rows where the username is in the list from users.csv
#         filtered_df = movie_df[movie_df['username'].isin(usernames)]

#         # Append dates to the corresponding user in the dictionary
#         for _, row in filtered_df.iterrows():
#             user_dates[row['username']].append(row['date'])
            

# # Convert the dictionary into a DataFrame
# result_df = pd.DataFrame({
#     'username': list(user_dates.keys()),
#     'count': users_df['count'],  
#     'dates': [list(pd.to_datetime(dates).year) for dates in user_dates.values()],
#     'range': [max(map(int, pd.to_datetime(dates).year)) - min(map(int, pd.to_datetime(dates).year)) for dates in user_dates.values()],  # Range of years
#     'mean': [int(sum(map(int, pd.to_datetime(dates).year)) / len(dates)) for dates in user_dates.values()]
# })

# result_df.to_csv(output_file, index=False)  # Save the result to a CSV file

import matplotlib.pyplot as plt
result_df = pd.read_csv(output_file)

bars = 10
plt.hist(result_df['range'], bins=bars, edgecolor='black')
plt.ylabel('Ratings')
plt.xlabel('Year')
plt.title('Ratings per Year Frequency Histogram')
plt.show()

bars = 20
plt.hist(result_df['count'], bins=bars, edgecolor='black')
plt.ylabel('Ratings')
plt.xlabel('Users')
plt.xlim(4, 200)
plt.title('Ratings per User Frequency Histogram')
plt.show()
