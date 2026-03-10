"""Second-stage reranker over retrieval candidates.

The retrieval tower only has user/item embeddings -- cheap enough to score
the whole catalog. The ranker is allowed to use richer, more expensive
features (genre overlap, popularity, retrieval score itself) because it only
ever scores the shortlist retrieval already narrowed down, not the full
catalog. That split is the reason a two-stage system exists instead of one
model doing both jobs.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from data import GENRE_COLS


def build_features(pairs: pd.DataFrame, items: pd.DataFrame, item_pop: pd.Series, retrieval_scores: dict) -> pd.DataFrame:
    df = pairs.merge(items[["item_id"] + GENRE_COLS], on="item_id", how="left")
    df["popularity"] = df["item_id"].map(item_pop).fillna(0)
    df["retrieval_score"] = [
        retrieval_scores.get((u, i), 0.0) for u, i in zip(df["user_id"], df["item_id"])
    ]
    feature_cols = GENRE_COLS + ["popularity", "retrieval_score"]
    return df, feature_cols


def sample_negatives(train: pd.DataFrame, all_items: np.ndarray, n_per_positive: int = 4, seed: int = 0) -> pd.DataFrame:
    """Random negatives for the ranker training set -- unlike retrieval
    training, the ranker only ever needs to distinguish 'obviously bad'
    candidates from good ones since retrieval already filtered out most of
    the catalog, so uniform random negatives are sufficient here."""
    rng = np.random.default_rng(seed)
    pos = train[train["label"] == 1]
    seen = train.groupby("user_id")["item_id"].apply(set).to_dict()
    rows = []
    for user_id in pos["user_id"].unique():
        n_pos = (pos["user_id"] == user_id).sum()
        already = seen.get(user_id, set())
        candidates = rng.choice(all_items, size=min(len(all_items), n_pos * n_per_positive * 3), replace=False)
        negs = [c for c in candidates if c not in already][: n_pos * n_per_positive]
        rows.extend((user_id, item_id, 0) for item_id in negs)
    return pd.DataFrame(rows, columns=["user_id", "item_id", "label"])


def train_ranker(train_features: pd.DataFrame, feature_cols: list[str]) -> GradientBoostingClassifier:
    clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.1)
    clf.fit(train_features[feature_cols], train_features["label"])
    return clf
