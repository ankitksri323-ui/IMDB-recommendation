def read_and_engineer(path, top_dir_k=30, has_target=True):
    df = pd.read_csv(path)
    if has_target:
        X, y, meta = engineer_features(df, top_dir_k=top_dir_k)
        return X, y, df, meta
    else:
        # For movies and watchlist, there is no 'your_rating' column
        X, _, meta = engineer_features(df, top_dir_k=top_dir_k)
        return X, df, meta

def generate_recommendations(ratings_file, watchlist_file, movies_file, topk=10, alpha=0.1, l1_ratio=0.1):
    # Use has_target=True for ratings (labeled), False for movies/watchlist (unlabeled)
    X_ratings, y_ratings, ratings_df, _ = read_and_engineer(ratings_file, has_target=True)
    X_watchlist, watchlist_df, _ = read_and_engineer(watchlist_file, has_target=False)
    X_movies, movies_df, _ = read_and_engineer(movies_file, has_target=False)

    # Make sure indices are aligned
    exclude_ids = set(ratings_df.get("tconst", ratings_df.get("movieId", []))) | set(watchlist_df.get("tconst", watchlist_df.get("movieId", [])))
    movie_ids = movies_df.get("tconst", movies_df.get("movieId"))
    candidate_mask = ~movie_ids.isin(exclude_ids)
    candidate_df = movies_df.loc[candidate_mask].reset_index(drop=True)
    candidate_X = X_movies.loc[candidate_mask].reset_index(drop=True)
    
    # Train ElasticNet
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=5000)
    model.fit(X_ratings, y_ratings)
    pred = np.clip(model.predict(candidate_X), 1.0, 10.0)
    candidate_df["predicted_rating"] = pred

    recommendations = candidate_df.sort_values("predicted_rating", ascending=False).head(topk)
    return recommendations
