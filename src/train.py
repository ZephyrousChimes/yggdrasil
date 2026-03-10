"""End-to-end pipeline: data -> retrieval tower -> ranker -> offline eval ->
saved artifacts for serving.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import load_ratings, load_items, time_based_split
from retrieval import train_retrieval
from ranker import build_features, sample_negatives, train_ranker
from eval import evaluate_ranking

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
RETRIEVAL_K = 50   # candidates pulled by the retrieval tower
FINAL_K = 10        # final recommendations returned after reranking


def main():
    ratings = load_ratings()
    items = load_items()
    train, test = time_based_split(ratings)

    user_ids = sorted(ratings["user_id"].unique())
    item_ids = sorted(ratings["item_id"].unique())
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {it: i for i, it in enumerate(item_ids)}
    idx2item = {i: it for it, i in item2idx.items()}

    pos_train = train[train["label"] == 1]
    user_idx = torch.tensor(pos_train["user_id"].map(user2idx).values, dtype=torch.long)
    item_idx = torch.tensor(pos_train["item_id"].map(item2idx).values, dtype=torch.long)

    model = train_retrieval(
        user_idx, item_idx, len(user_ids), len(item_ids),
        dim=64, epochs=40, batch_size=256, lr=0.005,
    )
    model.eval()

    # ---- retrieval candidates for every user (cosine sim via normalized dot product) ----
    with torch.no_grad():
        item_embs = model.all_item_embeddings()  # [n_items, dim]
        seen_by_user = train.groupby("user_id")["item_id"].apply(set).to_dict()

        retrieval_scores = {}
        candidates_by_user = {}
        for u in user_ids:
            u_emb = model.user_embedding(user2idx[u])
            scores = item_embs @ u_emb
            top = torch.topk(scores, k=min(RETRIEVAL_K, len(item_ids)))
            cand_items = [idx2item[i.item()] for i in top.indices]
            already = seen_by_user.get(u, set())
            cand_items = [it for it in cand_items if it not in already][:RETRIEVAL_K]
            candidates_by_user[u] = cand_items
            for it, score in zip(cand_items, top.values.tolist()):
                retrieval_scores[(u, it)] = score

    # ---- ranker: train on positives + sampled negatives, features = genres + popularity + retrieval score ----
    item_pop = train[train["label"] == 1]["item_id"].value_counts()
    negatives = sample_negatives(train, np.array(item_ids))
    ranker_train_pairs = pd.concat(
        [pos_train[["user_id", "item_id", "label"]], negatives], ignore_index=True
    )
    ranker_train_feats, feature_cols = build_features(ranker_train_pairs, items, item_pop, retrieval_scores)
    ranker = train_ranker(ranker_train_feats, feature_cols)

    # ---- rerank each user's retrieval candidates with the trained ranker ----
    user_to_recs = {}
    for u, cand_items in candidates_by_user.items():
        if not cand_items:
            user_to_recs[u] = []
            continue
        cand_df = pd.DataFrame({"user_id": u, "item_id": cand_items})
        feats, _ = build_features(cand_df, items, item_pop, retrieval_scores)
        scores = ranker.predict_proba(feats[feature_cols])[:, 1]
        ranked = [it for _, it in sorted(zip(scores, cand_items), reverse=True)]
        user_to_recs[u] = ranked[:FINAL_K]

    # ---- offline evaluation against the held-out, time-split test set ----
    user_to_relevant = test[test["label"] == 1].groupby("user_id")["item_id"].apply(set).to_dict()
    retrieval_only_recs = {u: c[:FINAL_K] for u, c in candidates_by_user.items()}
    metrics_retrieval_only = evaluate_ranking(retrieval_only_recs, user_to_relevant, k=FINAL_K)
    metrics_full = evaluate_ranking(user_to_recs, user_to_relevant, k=FINAL_K)

    # cold-start baseline for comparison: global popularity, same for every user
    top_popular = [int(x) for x in item_pop.index[:FINAL_K].tolist()]
    popularity_recs = {u: top_popular for u in user_ids}
    metrics_popularity = evaluate_ranking(popularity_recs, user_to_relevant, k=FINAL_K)

    print("Popularity baseline:      ", metrics_popularity)
    print("Retrieval only (no rerank):", metrics_retrieval_only)
    print("Retrieval + ranker:        ", metrics_full)

    # ---- persist artifacts for serving ----
    ARTIFACT_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ARTIFACT_DIR / "retrieval_model.pt")
    with open(ARTIFACT_DIR / "ranker.pkl", "wb") as f:
        pickle.dump(ranker, f)
    with open(ARTIFACT_DIR / "meta.json", "w") as f:
        json.dump(
            {
                "user2idx": {str(k): v for k, v in user2idx.items()},
                "item2idx": {str(k): v for k, v in item2idx.items()},
                "n_users": len(user_ids),
                "n_items": len(item_ids),
                "dim": 64,
                "feature_cols": feature_cols,
                "top_popular_fallback": top_popular,
                "metrics": {
                    "popularity_baseline": metrics_popularity,
                    "retrieval_only": metrics_retrieval_only,
                    "retrieval_plus_ranker": metrics_full,
                },
            },
            f,
        )
    item_pop.to_json(ARTIFACT_DIR / "item_popularity.json")
    items.to_json(ARTIFACT_DIR / "items.json", orient="records")
    print(f"\nartifacts written to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
