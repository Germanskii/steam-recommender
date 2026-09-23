"""Reproducible observed-interaction holdout; no rating values leak into fitting."""

import numpy as np
import pandas as pd
from steam_rec.model import validate_matrix, predict, recommend


def split_observed(frame, fraction=0.2, seed=42):
    validate_matrix(frame)
    if not 0 < fraction < 1:
        raise ValueError("Holdout fraction must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    held = pd.DataFrame(False, index=frame.index, columns=frame.columns)
    for i in range(len(frame)):
        seen = np.flatnonzero(frame.iloc[i].notna().to_numpy())
        if len(seen) >= 2:
            count = min(len(seen) - 1, max(1, int(np.floor(len(seen) * fraction))))
            held.iloc[i, rng.choice(seen, count, replace=False)] = True
    if not held.to_numpy().any():
        raise ValueError("Evaluation needs at least one player with two observed games.")
    return frame.mask(held), held


def evaluate(frame, metric="cosine", threshold=0.2, user_weight=0.5, fraction=0.2, seed=42, k=5):
    train, held = split_observed(frame, fraction, seed)
    predictions = predict(train, metric, threshold, user_weight)
    mask = held.to_numpy()
    truth = frame.to_numpy()[mask]
    rows = []
    for method, scores in predictions.items():
        estimated = scores.to_numpy()[mask]
        recalls, ndcgs = [], []
        for user in frame.index[held.any(axis=1)]:
            relevant = set(frame.columns[held.loc[user]])
            ranked = recommend(train, scores, user, k)["Game"].tolist()
            hits = np.array([game in relevant for game in ranked], dtype=float)
            dcg = float((hits / np.log2(np.arange(len(hits)) + 2)).sum())
            ideal = float((1 / np.log2(np.arange(min(k, len(relevant))) + 2)).sum())
            recalls.append(len(relevant.intersection(ranked)) / len(relevant))
            ndcgs.append(dcg / ideal)
        rows.append(
            {
                "Method": method,
                "RMSE": float(np.sqrt(np.mean((truth - estimated) ** 2))),
                "MAE": float(np.abs(truth - estimated).mean()),
                f"Recall@{k}": float(np.mean(recalls)),
                f"NDCG@{k}": float(np.mean(ndcgs)),
            }
        )
    details = {
        "seed": seed,
        "fraction": fraction,
        "held_interactions": int(mask.sum()),
        "evaluated_players": int(held.any(axis=1).sum()),
        "cold_items_in_train": int(train.isna().all().sum()),
        "similarity": metric,
        "threshold": threshold,
        "user_weight": user_weight,
        "ranking_k": k,
    }
    return pd.DataFrame(rows), details
