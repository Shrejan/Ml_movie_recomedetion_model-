import logging
import numpy as np

# Compute once at startup
logging.info("Computing embedding norms...")

embeddings = embeddings.astype(np.float32)

embedding_norms = np.linalg.norm(
    embeddings,
    axis=1
)

logging.info(
    f"Embedding norms shape: {embedding_norms.shape}"
)


def recommend_movies(user_movies, top_n=10):

    logging.info("========================================")
    logging.info("recommend_movies() started")
    logging.info(f"Input movies: {user_movies}")

    ratings_map = {
        movie["imdbId"]: movie["rating"]
        for movie in user_movies
    }

    logging.info(f"ratings_map: {ratings_map}")

    rows = metadata[
        metadata["imdbId"].isin(ratings_map.keys())
    ].copy()

    logging.info(f"Matched rows: {len(rows)}")

    if rows.empty:
        raise ValueError(
            "None of the provided imdbIds were found."
        )

    watched_indices = rows.index.to_numpy()

    logging.info(
        f"watched_indices shape: {watched_indices.shape}"
    )

    watched_movie_matrix = embeddings[
        watched_indices
    ]

    logging.info(
        f"watched_movie_matrix shape: {watched_movie_matrix.shape}"
    )

    weights = rows["imdbId"].map(
        ratings_map
    ).values.astype(np.float32)

    logging.info(f"weights: {weights}")

    # Build user vector
    logging.info("Building user vector")

    user_vector = np.average(
        watched_movie_matrix,
        axis=0,
        weights=weights
    ).astype(np.float32)

    logging.info(
        f"user_vector shape: {user_vector.shape}"
    )

    logging.info(
        f"user_vector dtype: {user_vector.dtype}"
    )

    # Cosine similarity using NumPy
    logging.info("Computing user norm")

    user_norm = np.linalg.norm(
        user_vector
    )

    logging.info(
        f"user_norm: {user_norm}"
    )

    logging.info("Computing similarities")

    similarities = (
        embeddings @ user_vector
    ) / (
        embedding_norms * user_norm + 1e-8
    )

    logging.info("Similarity computation completed")

    logging.info(
        f"similarities shape: {similarities.shape}"
    )

    logging.info(
        f"max similarity: {similarities.max()}"
    )

    logging.info(
        f"min similarity: {similarities.min()}"
    )

    logging.info("Sorting similarities")

    top_idx = np.argsort(
        similarities
    )[::-1]

    watched_ids = set(
        ratings_map.keys()
    )

    recommendations = []

    logging.info(
        "Building recommendation list"
    )

    for idx in top_idx:

        movie = metadata.iloc[idx]

        if movie["imdbId"] in watched_ids:
            continue

        recommendations.append(
            {
                "movie_title": movie["movie_title"],
                "imdbId": int(movie["imdbId"]),
                "similarity": float(
                    similarities[idx]
                ),
            }
        )

        if len(recommendations) >= top_n:
            break

    logging.info(
        f"Generated {len(recommendations)} recommendations"
    )

    if recommendations:
        logging.info(
            f"Top recommendation: {recommendations[0]}"
        )

    logging.info(
        "recommend_movies() completed"
    )

    logging.info("========================================")

    return recommendations
