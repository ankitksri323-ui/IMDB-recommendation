# combined.py
import sys, os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet

# add path for local scripts
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts", "training"))

from elasticnet_recommender import make_features, scale_features  
from baseline_recommender import recommend_baseline

RANDOM_STATE = 42

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_models(ratings_file, movies_file, test_size=0.2):
    ratings = pd.read_csv(ratings_file)
    movies = pd.read_csv(movies_file)

    # Train/test split
    train, test = train_test_split(ratings, test_size=test_size, random_state=RANDOM_STATE)

    # Baseline 
    global_mean = train["rating"].mean()
    movie_means = train.groupby("movieId")["rating"].mean()

    baseline_preds = test["movieId"].map(movie_means).fillna(global_mean)
    baseline_true = test["rating"]

    baseline_rmse = rmse(baseline_true, baseline_preds)
    baseline_mae = mean_absolute_error(baseline_true, baseline_preds)
    baseline_r2 = r2_score(baseline_true, baseline_preds)

    # ========== ElasticNet (with enriched features) ==========
    # Merge enriched movie features
    train = train.merge(movies, on="movieId", how="left")
    test = test.merge(movies, on="movieId", how="left")

    # Add baseline mean as explicit feature
    train["baseline_mean"] = train["movieId"].map(movie_means).fillna(global_mean)
    test["baseline_mean"] = test["movieId"].map(movie_means).fillna(global_mean)

    # Numeric columns to scale
    numeric_cols = ["year", "avg_rating", "rating_count", "rating_std",
                    "user_mean", "user_count", "user_std", "baseline_mean"]

    # One-hot encode genres
    genres_tokens = train["genres"].fillna("").apply(lambda s: [g.strip() for g in str(s).split("|") if g.strip()])
    test_genres_tokens = test["genres"].fillna("").apply(lambda s: [g.strip() for g in str(s).split("|") if g.strip()])

    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    train_genres = pd.DataFrame(
        mlb.fit_transform(genres_tokens), columns=[f"g_{g}" for g in mlb.classes_], index=train.index
    )
    test_genres = pd.DataFrame(
        mlb.transform(test_genres_tokens), columns=[f"g_{g}" for g in mlb.classes_], index=test.index
    )

    # Include tag columns directly
    tag_cols = [c for c in movies.columns if c.startswith("tag_")]

    # Final matrices
    X_train = pd.concat([train[numeric_cols], train_genres, train[tag_cols]], axis=1).fillna(0)
    X_test = pd.concat([test[numeric_cols], test_genres, test[tag_cols]], axis=1).fillna(0)

    y_train = train["rating"]
    y_test = test["rating"]

    # Align columns
    all_cols = sorted(set(X_train.columns) | set(X_test.columns))
    X_train = X_train.reindex(columns=all_cols, fill_value=0.0)
    X_test = X_test.reindex(columns=all_cols, fill_value=0.0)


    X_train[numeric_cols] = X_train[numeric_cols].astype(float)
    X_test[numeric_cols] = X_test[numeric_cols].astype(float)

    mu = X_train[numeric_cols].mean()
    sd = X_train[numeric_cols].std().replace(0, 1.0)
    X_train.loc[:, numeric_cols] = (X_train[numeric_cols] - mu) / sd
    X_test.loc[:, numeric_cols] = (X_test[numeric_cols] - mu) / sd

    # Grid search ElasticNet
    param_grid = {
        "alpha": [0.001, 0.01, 0.1, 1.0],
        "l1_ratio": [0.1, 0.5, 0.9]
    }
    grid = GridSearchCV(
        ElasticNet(random_state=RANDOM_STATE, max_iter=5000),
        param_grid,
        scoring="neg_mean_squared_error",
        cv=3,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    elastic_preds = np.clip(best_model.predict(X_test), 0.5, 5.0)

    elastic_rmse = rmse(y_test, elastic_preds)
    elastic_mae = mean_absolute_error(y_test, elastic_preds)
    elastic_r2 = r2_score(y_test, elastic_preds)

    # ----------------------------
    # Print results
    # ----------------------------
    print("Model Comparison (on test set, enriched features)")
    print("="*60)
    print(f"Baseline   → RMSE: {baseline_rmse:.4f}, MAE: {baseline_mae:.4f}, R²: {baseline_r2:.4f}")
    print(f"ElasticNet (enriched) → RMSE: {elastic_rmse:.4f}, MAE: {elastic_mae:.4f}, R²: {elastic_r2:.4f}")
    print("-"*60)
    print("Best ElasticNet Params:", grid.best_params_)

if __name__ == "__main__":
    evaluate_models("data/enriched/ratings_enriched.csv", "data/enriched/movies_enriched.csv")
