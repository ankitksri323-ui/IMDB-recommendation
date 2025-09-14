import pandas as pd
import random
import os

# Paths
movies_file = "data/raw/movies.csv"
watchlist_file = "data/raw/watchlist.csv"

# Ensure output folder exists
os.makedirs(os.path.dirname(watchlist_file), exist_ok=True)

# Read movies
movies = pd.read_csv(movies_file)

# Randomly select 25 movies for watchlist
watchlist_sample = movies.sample(n=25, random_state=42)

# Keep only movieId column
watchlist = watchlist_sample[["movieId"]]

# Save to CSV
watchlist.to_csv(watchlist_file, index=False)
print(f"Watchlist saved to {watchlist_file}")
