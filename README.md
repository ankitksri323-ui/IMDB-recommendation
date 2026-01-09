# IMDb Movie Recommender System

This project implements a movie recommender system using:
- A **Baseline model** (mean rating)
- An **ElasticNet model** with enriched content-based features

The goal is to compare a simple baseline against a regularized linear model for rating prediction and recommendation quality.

---

## Models

### 1. Baseline Recommender
- Predicts ratings using global / per-movie mean ratings
- Serves as a strong reference model

### 2. ElasticNet Recommender
- Uses enriched features such as:
  - Movie metadata (genres, year)
  - Aggregated rating statistics
- Trained using ElasticNet regularization to balance bias–variance tradeoff

---

## Evaluation Metrics

The models were evaluated using standard regression metrics:

| Model                    | RMSE  | MAE   | R²    |
|--------------------------|-------|-------|-------|
| Baseline                 | 0.9827 | 0.7604 | 0.1221 |
| ElasticNet (enriched)    | **0.8901** | **0.6810** | **0.2797** |

**Observation:**  
ElasticNet outperforms the baseline across all metrics, achieving lower error (RMSE, MAE) and higher explained variance (R²).

---

## How to Run

### 1. Prepare data
```bash
python download_and_convert_movielens.py
python prepare_data.py
