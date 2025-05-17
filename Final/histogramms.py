import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def ensure_users_exist():
    path = "ModifiedData/users.csv"
    if not os.path.exists(path):
        print("⚠ Required file 'ModifiedData/users.csv' not found.")
        print("→ Generating it now using user_matrix module...")
        try:
            import Final.user_matrix as user_matrix
            user_matrix.main()
        except Exception as e:
            print(f"✗ Failed to generate user matrix: {e}")
            exit(1)
    else:
        print("✓ Found users.csv.")

def load_review_dates(users_file, reviews_folder):
    users_df = pd.read_csv(users_file)
    usernames = set(users_df['username'])

    user_dates = {username: [] for username in usernames}

    for file in os.listdir(reviews_folder):
        if file.endswith(".csv"):
            movie_path = os.path.join(reviews_folder, file)
            movie_df = pd.read_csv(movie_path)
            filtered_df = movie_df[movie_df['username'].isin(usernames)]

            for _, row in filtered_df.iterrows():
                try:
                    user_dates[row['username']].append(pd.to_datetime(row['date']))
                except Exception:
                    continue  # Skip malformed dates

    # Build final DataFrame
    final_df = pd.DataFrame({
        'username': list(user_dates.keys()),
        'count': [len(dates) for dates in user_dates.values()],
        'range': [(max(dates) - min(dates)).days if dates else 0 for dates in user_dates.values()],
        'mean_year': [int(sum(d.year for d in dates) / len(dates)) if dates else 0 for dates in user_dates.values()]
    })

    final_df.to_csv("ModifiedData/user_dates.csv", index=False)
    return final_df

def plot_histograms(df, range_bins=10, count_bins=20, show=True):
    # Plot rating range histogram
    plt.figure(figsize=(8, 5))
    plt.hist(df['range'], bins=range_bins, edgecolor='black')
    plt.ylabel('Number of Users')
    plt.xlabel('Days Between First and Last Rating')
    plt.title('Rating Range per User')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if show:
        plt.show()

    # Plot count histogram
    plt.figure(figsize=(8, 5))
    plt.hist(df['count'], bins=count_bins, edgecolor='black')
    plt.ylabel('Number of Users')
    plt.xlabel('Ratings per User')
    plt.title('Number of Ratings per User')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if show:
        plt.show()

def create_histogramms():
    print("=== User Rating Histograms ===")
    try:
        range_bins = int(input("Enter number of bins for rating range histogram (default 10): ") or "10")
        count_bins = int(input("Enter number of bins for count histogram (default 20): ") or "20")
    except ValueError:
        print("Invalid input. Using defaults: 10 and 20 bins.")
        range_bins, count_bins = 10, 20

    display = input("Show plots? (y/n, default y): ").strip().lower()
    show = display != "n"

    ensure_users_exist()
    df = load_review_dates("ModifiedData/users.csv", "Dataset/2_reviews_per_movie_raw")
    print("✓ Data processed. Plotting histograms...")
    plot_histograms(df, range_bins=range_bins, count_bins=count_bins, show=show)

if __name__ == "__main__":
    create_histogramms()
