#!/usr/bin/env python3
# File: elasticnet_recommender.py
# Purpose: ElasticNet-based MovieLens recommender with enriched features

import numpy as np
import pandas as pd
import typer
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import MultiLabelBinarizer

RANDOM_STATE = 42

def make_features(df, movies_df, movie_means=None, global_mean=None):
    """Create feature matrix X and target y from DataFrame (uses enriched movies)."""
    y = df["rating"].astype(float).values if "rating" in df else None

    # Baseline mean rating as feature
    if movie_means is not None:
        df["baseline_mean"] = df["movieId"].map(movie_means).fillna(global_mean)
    else:
        df["baseline_mean"] = df["avg_rating"].fillna(global_mean)

    # Genres multi-hot
    genres_tokens = df["genres"].fillna("").apply(lambda s: [g.strip() for g in str(s).split("|") if g.strip()])
    mlb = MultiLabelBinarizer()
    genres_mh = pd.DataFrame(
        mlb.fit_transform(genres_tokens),
        columns=[f"g_{g}" for g in mlb.classes_],
        index=df.index
    )

    # Tag features (already multi-hot in enriched movies)
    tag_cols = [c for c in movies_df.columns if c.startswith("tag_")]
    df_tags = df[tag_cols] if tag_cols else pd.DataFrame(index=df.index)

    # Numeric features
    numeric_cols = [
        "year", "avg_rating", "rating_count", "rating_std",
        "user_mean", "user_count", "user_std", "baseline_mean"
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    X = pd.concat([genres_mh, df[numeric_cols].astype(float), df_tags], axis=1).fillna(0.0)

    return X, y, numeric_cols

def scale_features(X_train, X_pred, numeric_cols):
    """Standardize numeric columns."""
    if numeric_cols:
        mu = X_train[numeric_cols].mean()
        sd = X_train[numeric_cols].std().replace(0, 1.0)
        X_train.loc[:, numeric_cols] = (X_train[numeric_cols] - mu) / sd
        X_pred.loc[:, numeric_cols] = (X_pred[numeric_cols] - mu) / sd
    return X_train, X_pred

def generate_recommendations(ratings_file, watchlist_file, movies_file, topk=10, alpha=0.1, l1_ratio=0.1):
    """Generate top-N movie recommendations using enriched features."""
    ratings_df = pd.read_csv(str(ratings_file))
    watchlist_df = pd.read_csv(str(watchlist_file))
    movies_df = pd.read_csv(str(movies_file))

    # Compute movie mean ratings for baseline feature
    global_mean = ratings_df["rating"].mean()
    movie_means = ratings_df.groupby("movieId")["rating"].mean()

    # Exclude already rated or in watchlist
    exclude_ids = set(ratings_df["movieId"]).union(set(watchlist_df["movieId"]))
    candidate_df = movies_df[~movies_df["movieId"].isin(exclude_ids)].copy()

    # Merge features into ratings for training
    ratings_df = ratings_df.merge(movies_df, on="movieId", how="left")

    # Training features
    X_train, y_train, numeric_cols = make_features(ratings_df, movies_df, movie_means, global_mean)

    # Prediction features
    candidate_df = candidate_df.merge(ratings_df[["movieId", "user_mean", "user_count", "user_std"]].drop_duplicates("movieId"),
                                      on="movieId", how="left")
    X_pred, _, _ = make_features(candidate_df, movies_df, movie_means, global_mean)

    # Align columns
    all_cols = sorted(set(X_train.columns) | set(X_pred.columns))
    X_train = X_train.reindex(columns=all_cols, fill_value=0.0)
    X_pred = X_pred.reindex(columns=all_cols, fill_value=0.0)

    # Scale numeric columns
    numeric_cols = [c for c in numeric_cols if c in X_train.columns]
    X_train, X_pred = scale_features(X_train, X_pred, numeric_cols)

    # Train ElasticNet
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=RANDOM_STATE, max_iter=5000)
    model.fit(X_train, y_train)

    # Predictions
    preds = np.clip(model.predict(X_pred), 0.5, 5.0)  # MovieLens rating scale 0.5–5
    candidate_df["predicted_rating"] = preds

    # Sort and take top-k
    recommendations = candidate_df.sort_values("predicted_rating", ascending=False).head(topk)

    return recommendations

app = typer.Typer()

@app.command()
def run(
    ratings_file: str = typer.Option(..., help="Path to enriched ratings CSV"),
    watchlist_file: str = typer.Option(..., help="Path to watchlist CSV"),
    movies_file: str = typer.Option(..., help="Path to enriched movies CSV"),
    topk: int = typer.Option(10, help="Number of movies to recommend"),
    alpha: float = typer.Option(0.1, help="ElasticNet alpha"),
    l1_ratio: float = typer.Option(0.1, help="ElasticNet l1_ratio")
):
    try:
        recs = generate_recommendations(ratings_file, watchlist_file, movies_file, topk, alpha, l1_ratio)
        print(f"\nTop {topk} Recommendations (ElasticNet with enriched features):\n" + "-"*70)
        for i, (_, row) in enumerate(recs.iterrows(), 1):
            title = row.get("title", "Unknown")
            year = row.get("year", "")
            genres = row.get("genres", "")
            print(f"{i}. {title} ({year}) - Predicted: {row['predicted_rating']:.2f} | {genres}")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    app()
