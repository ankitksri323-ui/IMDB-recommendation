from scripts.training.baseline_recommender import recommend_baseline
from scripts.training.elasticnet_recommender import generate_recommendations

RATINGS_FILE = "data/raw/ratings.csv"
WATCHLIST_FILE = "data/raw/watchlist.csv"
MOVIES_RAW_FILE = "data/raw/movies.csv"
MOVIES_ENRICHED_FILE = "data/enriched/movies_enriched.csv"
RATINGS_ENRICHED_FILE = "data/enriched/ratings_enriched.csv"

TOP_K = 10


def run_baseline():
    print("\n🎯 BASELINE RECOMMENDATIONS (Mean Rating)")
    print("=" * 50)

    recs = recommend_baseline(
        ratings_file=RATINGS_FILE,
        watchlist_file=WATCHLIST_FILE,
        movies_file=MOVIES_RAW_FILE,
        topk=TOP_K
    )

    for i, (_, row) in enumerate(recs.iterrows(), 1):
        print(f"{i:2d}. {row['title']} → ⭐ {row['pred']:.2f}")


def run_elasticnet():
    print("\n🎯 ELASTICNET RECOMMENDATIONS (Enriched Features)")
    print("=" * 50)

    recs = generate_recommendations(
        ratings_file=RATINGS_ENRICHED_FILE,
        watchlist_file=WATCHLIST_FILE,
        movies_file=MOVIES_ENRICHED_FILE,
        topk=TOP_K
    )

    for i, (_, row) in enumerate(recs.iterrows(), 1):
        title = row.get("title", "Unknown")
        year = row.get("year", "")
        print(f"{i:2d}. {title} ({year}) → ⭐ {row['predicted_rating']:.2f}")


if __name__ == "__main__":
    run_baseline()
    run_elasticnet()
