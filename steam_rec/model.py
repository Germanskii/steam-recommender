"""Explicit, small-scale neighborhood collaborative filtering."""

import numpy as np
import pandas as pd

METHODS = ["User-based", "Item-based", "Hybrid", "Item-mean baseline"]
SIMILARITIES = ["cosine", "jaccard", "pearson", "mixed"]


def validate_matrix(frame):
    if frame.empty or not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError("Matrix must be nonempty with unique players and games.")
    if frame.shape[0] > 601 or frame.shape[1] > 100:
        raise ValueError("Dense demonstration is limited to 601 players and 100 games.")
    values = frame.to_numpy(dtype=float)
    observed = values[~np.isnan(values)]
    if (
        not len(observed)
        or not np.isfinite(observed).all()
        or ((observed < 0.5) | (observed > 5)).any()
    ):
        raise ValueError(
            "Observed scores must be finite and between 0.5 and 5; missing values are NaN."
        )
    return values


def similarity(values, metric="cosine"):
    if metric not in SIMILARITIES:
        raise ValueError("Unknown similarity metric.")
    present = ~np.isnan(values)
    filled = np.nan_to_num(values, nan=0.0)
    dot = filled @ filled.T
    norms = np.linalg.norm(filled, axis=1)
    cosine = np.divide(
        dot, np.outer(norms, norms), out=np.zeros_like(dot), where=np.outer(norms, norms) > 0
    )
    if metric == "cosine":
        result = cosine
    elif metric == "jaccard":
        mask = present.astype(float)
        intersection = mask @ mask.T
        union = mask.sum(1)[:, None] + mask.sum(1)[None, :] - intersection
        result = np.divide(intersection, union, out=np.zeros_like(dot), where=union > 0)
    else:
        # Pearson uses only jointly observed entries; missing is not a zero rating.
        mask = present.astype(float)
        n = mask @ mask.T
        sums = filled @ mask.T
        squares = (filled**2) @ mask.T
        correction = np.divide(sums * sums.T, n, out=np.zeros_like(dot), where=n > 0)
        variance = squares - np.divide(sums**2, n, out=np.zeros_like(dot), where=n > 0)
        denominator = np.sqrt(np.maximum(variance, 0) * np.maximum(variance.T, 0))
        pearson = np.divide(
            dot - correction,
            denominator,
            out=np.zeros_like(dot),
            where=(n >= 2) & (denominator > 1e-10),
        ).clip(0, 1)
        result = pearson if metric == "pearson" else 0.5 * cosine + 0.5 * pearson
    np.fill_diagonal(result, 0)
    return result.clip(0, 1)


def predict(frame, metric="cosine", threshold=0.2, user_weight=0.5):
    values = validate_matrix(frame)
    if (
        not np.isfinite([threshold, user_weight]).all()
        or not 0 <= threshold <= 1
        or not 0 <= user_weight <= 1
    ):
        raise ValueError("Threshold and user weight must be between 0 and 1.")
    mask = ~np.isnan(values)
    filled = np.nan_to_num(values, nan=0)
    global_mean = filled.sum() / mask.sum()
    user_mean = np.divide(
        filled.sum(1), mask.sum(1), out=np.full(len(frame), global_mean), where=mask.sum(1) > 0
    )
    item_mean = np.divide(
        filled.sum(0), mask.sum(0), out=np.full(frame.shape[1], global_mean), where=mask.sum(0) > 0
    )
    users = similarity(values, metric)
    items = similarity(values.T, metric)
    users[users <= threshold] = 0
    items[items <= threshold] = 0
    user_num = users @ np.where(mask, values - user_mean[:, None], 0)
    user_den = users @ mask.astype(float)
    user_predictions = user_mean[:, None] + np.divide(
        user_num, user_den, out=np.zeros_like(values), where=user_den > 0
    )
    item_num = np.where(mask, values - item_mean[None, :], 0) @ items
    item_den = mask.astype(float) @ items
    item_predictions = item_mean[None, :] + np.divide(
        item_num, item_den, out=np.zeros_like(values), where=item_den > 0
    )
    # No contributing neighbors: user mean / item mean, never uninitialized memory.
    arrays = [
        user_predictions,
        item_predictions,
        user_weight * user_predictions + (1 - user_weight) * item_predictions,
        np.broadcast_to(item_mean, values.shape),
    ]
    return {
        name: pd.DataFrame(array.clip(0.5, 5), index=frame.index, columns=frame.columns)
        for name, array in zip(METHODS, arrays)
    }


def recommend(frame, scores, user, top_n=10):
    if not scores.index.equals(frame.index) or not scores.columns.equals(frame.columns):
        raise ValueError("Predictions must align with the source matrix.")
    if user not in frame.index or not 1 <= top_n <= 100:
        raise ValueError("Choose a known player and top_n between 1 and 100.")
    unseen = frame.loc[user].isna()
    ranked = scores.loc[user, unseen].sort_values(ascending=False, kind="stable").head(top_n)
    return ranked.rename_axis("Game").reset_index(name="Estimated score")
