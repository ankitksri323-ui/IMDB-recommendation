# scripts/training/baseline_recommender.py
import argparse
import pandas as pd

def recommend_baseline(ratings_file, watchlist_file, movies_file, topk=10):
    # Load ratings, watchlist, movies
    ratings = pd.read_csv(ratings_file)
    watchlist = pd.read_csv(watchlist_file)
    movies = pd.read_csv(movies_file)

    # Use global mean rating per movie (non-personalized baseline)
    movie_means = ratings.groupby("movieId")["rating"].mean().reset_index()
    movies = movies.merge(movie_means, on="movieId", how="left")

    # Fill missing with overall mean
    overall_mean = ratings["rating"].mean()
    movies["pred"] = movies["rating"].fillna(overall_mean)

    # Exclude rated + watchlist movies
    exclude_ids = set(ratings["movieId"]).union(set(watchlist["movieId"]))
    recs = movies[~movies["movieId"].isin(exclude_ids)]

    # Sort by predicted (mean rating) desc
    recs = recs.sort_values("pred", ascending=False).head(topk)

    return recs[["movieId", "title", "genres", "pred"]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Mean Rating Recommender")
    parser.add_argument("--ratings-file", required=True)
    parser.add_argument("--watchlist-file", required=True)
    parser.add_argument("--movies-file", required=True)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    recs = recommend_baseline(args.ratings_file, args.watchlist_file, args.movies_file, args.topk)

    print("🎬 Baseline Recommendations (Mean User Rating)")
    print("="*50)
    for i, row in recs.iterrows():
        print(f"{row['title']} - Predicted {row['pred']:.2f} | {row['genres']}")
