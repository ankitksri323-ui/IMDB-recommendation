#!/usr/bin/env python3
# File: elasticnet_recommender.py
# Purpose: ElasticNet-based MovieLens recommender (utility module)

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import MultiLabelBinarizer

RANDOM_STATE = 42
RESULTS_FILE = "results/comparison_results.csv"


def load_best_params(default_alpha=0.1, default_l1=0.1):
    """Load the most recent best alpha/l1_ratio from results CSV."""
    if os.path.exists(RESULTS_FILE):
        try:
            df = pd.read_csv(RESULTS_FILE)
            if "best_alpha" in df.columns and "best_l1_ratio" in df.columns:
                last_row = df.iloc[-1]
                return float(last_row["best_alpha"]), float(last_row["best_l1_ratio"])
        except Exception:
            pass
    return default_alpha, default_l1


def make_features(df, movies_df, movie_means=None, global_mean=None):
    """Create feature matrix X and target y."""
    y = df["rating"].astype(float).values if "rating" in df else None

    # Baseline mean feature
    if movie_means is not None:
        df["baseline_mean"] = df["movieId"].map(movie_means).fillna(global_mean)
    else:
        df["baseline_mean"] = df["avg_rating"].fillna(global_mean)

    # Genres (multi-hot)
    genres_tokens = df["genres"].fillna("").apply(
        lambda s: [g.strip() for g in str(s).split("|") if g.strip()]
    )
    mlb = MultiLabelBinarizer()
    genres_mh = pd.DataFrame(
        mlb.fit_transform(genres_tokens),
        columns=[f"g_{g}" for g in mlb.classes_],
        index=df.index
    )

    # Tag features
    tag_cols = [c for c in movies_df.columns if c.startswith("tag_")]
    tag_df = df[tag_cols] if tag_cols else pd.DataFrame(index=df.index)

    # Numeric features
    numeric_cols = [
        "year", "avg_rating", "rating_count", "rating_std",
        "user_mean", "user_count", "user_std", "baseline_mean"
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    X = pd.concat(
        [genres_mh, df[numeric_cols].astype(float), tag_df],
        axis=1
    ).fillna(0.0)

    return X, y, numeric_cols


def scale_features(X_train, X_pred, numeric_cols):
    """Standardize numeric columns using training statistics."""
    if numeric_cols:
        mu = X_train[numeric_cols].mean()
        sd = X_train[numeric_cols].std().replace(0, 1.0)
        X_train.loc[:, numeric_cols] = (X_train[numeric_cols] - mu) / sd
        X_pred.loc[:, numeric_cols] = (X_pred[numeric_cols] - mu) / sd
    return X_train, X_pred


def generate_recommendations(
    ratings_file,
    watchlist_file,
    movies_file,
    topk=10,
    alpha=None,
    l1_ratio=None
):
    """Generate top-N movie recommendations."""
    ratings_df = pd.read_csv(ratings_file)
    watchlist_df = pd.read_csv(watchlist_file)
    movies_df = pd.read_csv(movies_file)

    global_mean = ratings_df["rating"].mean()
    movie_means = ratings_df.groupby("movieId")["rating"].mean()

    # Exclude already rated / watchlisted movies
    exclude_ids = set(ratings_df["movieId"]).union(set(watchlist_df["movieId"]))
    candidate_df = movies_df[~movies_df["movieId"].isin(exclude_ids)].copy()

    # Merge movie features into ratings
    ratings_df = ratings_df.merge(movies_df, on="movieId", how="left")

    # User stats
    user_stats = ratings_df.groupby("movieId")["rating"].agg(
        user_mean="mean", user_count="count", user_std="std"
    ).reset_index()
    ratings_df = ratings_df.merge(user_stats, on="movieId", how="left")
    candidate_df = candidate_df.merge(user_stats, on="movieId", how="left")

    # Features
    X_train, y_train, numeric_cols = make_features(
        ratings_df, movies_df, movie_means, global_mean
    )
    X_pred, _, _ = make_features(
        candidate_df, movies_df, movie_means, global_mean
    )

    # Align columns
    all_cols = sorted(set(X_train.columns) | set(X_pred.columns))
    X_train = X_train.reindex(columns=all_cols, fill_value=0.0)
    X_pred = X_pred.reindex(columns=all_cols, fill_value=0.0)

    # Scale
    numeric_cols = [c for c in numeric_cols if c in X_train.columns]
    X_train, X_pred = scale_features(X_train, X_pred, numeric_cols)

    # Params
    if alpha is None or l1_ratio is None:
        alpha, l1_ratio = load_best_params()

    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=RANDOM_STATE,
        max_iter=5000
    )
    model.fit(X_train, y_train)

    preds = np.clip(model.predict(X_pred), 0.5, 5.0)
    candidate_df["predicted_rating"] = preds

    return candidate_df.sort_values(
        "predicted_rating", ascending=False
    ).head(topk)
