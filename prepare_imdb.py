#!/usr/bin/env python3
# File: prepare_imdb.py
# Purpose: Convert IMDb .tsv.gz into clean CSV for recommender

import pandas as pd

def prepare_imdb(basics_file="title.basics.tsv.gz", ratings_file="title.ratings.tsv.gz", output="imdb_movies.csv"):
    print("Loading IMDb basics...")
    basics = pd.read_csv(basics_file, sep="\t", dtype=str, na_values="\\N")

    print("Loading IMDb ratings...")
    ratings = pd.read_csv(ratings_file, sep="\t", dtype=str, na_values="\\N")

    # Keep only movies
    movies = basics[basics["titleType"] == "movie"].copy()

    # Merge ratings
    movies = movies.merge(ratings, on="tconst", how="inner")

    # Rename to match recommender expectations
    movies = movies.rename(columns={
        "primaryTitle": "Title",
        "startYear": "Year",
        "genres": "Genres",
        "averageRating": "imdb",
        "numVotes": "votes"
    })

    # Keep relevant columns
    keep_cols = ["tconst", "Title", "Year", "Genres", "imdb", "votes"]
    movies = movies[keep_cols]

    # Save
    movies.to_csv(output, index=False)
    print(f"Saved {output} with {len(movies)} movies")

if __name__ == "__main__":
    prepare_imdb()
