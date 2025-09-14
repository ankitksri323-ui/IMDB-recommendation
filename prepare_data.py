# prepare_data.py
import pandas as pd
import numpy as np

def prepare_data(movies_file, ratings_file, tags_file, out_dir="data/enriched"):
    # Load CSVs
    movies = pd.read_csv(movies_file)
    ratings = pd.read_csv(ratings_file)
    tags = pd.read_csv(tags_file)

    # ----------------------------
    # Movie-level features
    # ----------------------------
    movie_stats = ratings.groupby("movieId")["rating"].agg(
        avg_rating="mean",
        rating_count="count",
        rating_std="std"
    ).reset_index()
    movies = movies.merge(movie_stats, on="movieId", how="left")

    # Extract year from title
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)

    # ----------------------------
    # User-level features
    # ----------------------------
    user_stats = ratings.groupby("userId")["rating"].agg(
        user_mean="mean",
        user_count="count",
        user_std="std"
    ).reset_index()
    ratings = ratings.merge(user_stats, on="userId", how="left")

    # ----------------------------
    # Tag features (multi-hot encoding)
    # ----------------------------
    # Keep only top-N most frequent tags to avoid too many columns
    top_tags = tags["tag"].value_counts().head(50).index  # top 50 tags
    tags_filtered = tags[tags["tag"].isin(top_tags)]

    tags_pivot = (
        tags_filtered
        .assign(val=1)
        .pivot_table(index="movieId", columns="tag", values="val", fill_value=0)
        .add_prefix("tag_")
    )
    movies = movies.merge(tags_pivot, on="movieId", how="left").fillna(0)

    # ----------------------------
    # Save enriched CSVs
    # ----------------------------
    import os
    os.makedirs(out_dir, exist_ok=True)

    movies_out = f"{out_dir}/movies_enriched.csv"
    ratings_out = f"{out_dir}/ratings_enriched.csv"

    movies.to_csv(movies_out, index=False)
    ratings.to_csv(ratings_out, index=False)

    print(f" Saved enriched movies → {movies_out}")
    print(f" Saved enriched ratings → {ratings_out}")

if __name__ == "__main__":
    prepare_data(
        movies_file="data/raw/movies.csv",
        ratings_file="data/raw/ratings.csv",
        tags_file="data/raw/tags.csv",
        out_dir="data/enriched"
    )
