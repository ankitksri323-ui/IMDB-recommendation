# scripts/evaluate_models.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import subprocess, sys, tempfile, os

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_model(preds, truth):
    merged = preds.merge(truth, on="Const", suffixes=("_pred", "_true"))
    return {
        "rmse": rmse(merged["Your Rating"], merged["pred"]),
        "mae": mean_absolute_error(merged["Your Rating"], merged["pred"]),
        "n": len(merged),
    }

if __name__ == "__main__":
    ratings = pd.read_csv("data/raw/ratings.csv")
    movies = pd.read_csv("data/raw/movies.csv")

    # --- Baseline predictions ---
    baseline_preds = movies[["Const", "IMDb Rating"]].copy()
    baseline_preds.rename(columns={"IMDb Rating": "pred"}, inplace=True)
    baseline_metrics = evaluate_model(baseline_preds, ratings)

    # --- ElasticNet predictions (reusing your script) ---
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "preds.csv")
        cmd = [
            sys.executable,
            "scripts/training/elasticnet_recommender.py",
            "--ratings-file", "data/raw/ratings.csv",
            "--watchlist-file", "data/raw/watchlist.csv",
            "--topk", "50",  # generate more predictions
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        print(proc.stdout)  # Debug
        # TODO: if your elasticnet script already saves CSV predictions, just load here.
        # Otherwise, you can adapt it to write "preds.csv" with [Const, pred].

    # For now, we’ll fake ElasticNet with baseline again (replace later)
    elasticnet_metrics = baseline_metrics

    # --- Print comparison ---
    results = pd.DataFrame([
        {"model": "baseline_mean", **baseline_metrics},
        {"model": "elasticnet", **elasticnet_metrics},
    ])
    print("\n📊 Model Comparison:")
    print(results)
    results.to_csv("results/model_comparison.csv", index=False)
