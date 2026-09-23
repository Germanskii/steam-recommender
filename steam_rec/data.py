"""Local Steam-200k import and an original synthetic demonstration."""

import numpy as np
import pandas as pd

MAX_USERS = 600
MAX_ITEMS = 100


def demo_matrix(seed=42):
    """Invented players and games; not measured Steam behavior."""
    rng = np.random.default_rng(seed)
    games = [
        f"{genre} {name}"
        for genre in ["Orbit", "Rally", "Kingdom", "Puzzle"]
        for name in ["Quest", "Lab", "World", "Journey", "Arena", "Chronicles"]
    ]
    groups = np.arange(48) % 4
    scores = np.full((48, len(games)), np.nan)
    for user, group in enumerate(groups):
        favored = rng.choice(np.arange(group * 6, group * 6 + 6), size=4, replace=False)
        other = rng.choice(np.setdiff1d(np.arange(24), favored), size=4, replace=False)
        for item in np.r_[favored, other]:
            scores[user, item] = np.clip(4.4 if item // 6 == group else 1.8, 0.5, 5)
            scores[user, item] = np.clip(scores[user, item] + rng.normal(0, 0.3), 0.5, 5)
    return pd.DataFrame(scores, index=[f"demo-{i + 1:02}" for i in range(48)], columns=games)


def load_steam_csv(source, max_users=MAX_USERS, max_items=MAX_ITEMS):
    """Read the headerless five-column Steam-200k format, returning a bounded matrix.

    Hours are an implicit engagement proxy, not star ratings. The fixed transform
    does not learn a normalizing maximum from held-out values.
    """
    if not 2 <= max_users <= MAX_USERS or not 2 <= max_items <= MAX_ITEMS:
        raise ValueError("Limits must be 2–600 users and 2–100 games.")
    raw = pd.read_csv(source, header=None, encoding="utf-8-sig")
    if raw.shape[1] != 5:
        raise ValueError("Expected five headerless columns: user, game, action, hours, value.")
    raw.columns = ["user", "game", "action", "hours", "value"]
    if raw[["user", "game", "action", "hours"]].isna().any().any():
        raise ValueError("Input contains missing required fields.")
    if not raw["action"].isin(["play", "purchase"]).all():
        raise ValueError("Action must be play or purchase; remove the CSV header if present.")
    plays = raw[raw["action"] == "play"].copy()
    if plays.empty:
        raise ValueError("No play interactions found.")
    plays["hours"] = pd.to_numeric(plays["hours"], errors="raise")
    if not np.isfinite(plays["hours"]).all() or (plays["hours"] < 0).any():
        raise ValueError("Playtime must be finite and non-negative.")
    plays["user"] = plays["user"].astype(str)
    plays["game"] = plays["game"].astype(str).str.strip()
    if plays["game"].eq("").any():
        raise ValueError("Game names cannot be blank.")
    # Repeated records are treated as duplicate snapshots, not additive sessions.
    pairs = plays.groupby(["user", "game"], as_index=False)["hours"].max()
    items = (
        pairs.groupby("game")
        .size()
        .sort_values(ascending=False, kind="stable")
        .head(max_items)
        .index
    )
    pairs = pairs[pairs["game"].isin(items)]
    activity = pairs.groupby("user").size()
    users = (
        activity[activity >= 2].sort_values(ascending=False, kind="stable").head(max_users).index
    )
    pairs = pairs[pairs["user"].isin(users)].copy()
    if pairs["user"].nunique() < 2 or pairs["game"].nunique() < 2:
        raise ValueError("At least two players with two games each are required after filtering.")
    pairs["score"] = (0.5 + 4.5 * np.log1p(pairs["hours"]) / np.log1p(1000)).clip(0.5, 5)
    result = pairs.pivot(index="user", columns="game", values="score")
    # Stable aliases prevent accidental display/export of source account IDs.
    result.index = [f"player-{i + 1:04}" for i in range(len(result))]
    return result
