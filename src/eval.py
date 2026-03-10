"""Ranking metrics: Recall@K and NDCG@K.

Accuracy isn't used here on purpose -- this is a ranking task where *where*
a relevant item lands in the list matters, not just whether it's present
anywhere in the candidate pool.
"""
import numpy as np


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return np.nan
    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return np.nan
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking(user_to_recs: dict[int, list[int]], user_to_relevant: dict[int, set[int]], k: int = 10):
    recalls, ndcgs = [], []
    for user_id, relevant in user_to_relevant.items():
        recs = user_to_recs.get(user_id, [])
        recalls.append(recall_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))
    return {
        f"Recall@{k}": float(np.nanmean(recalls)),
        f"NDCG@{k}": float(np.nanmean(ndcgs)),
        "n_users_evaluated": len(user_to_relevant),
    }
