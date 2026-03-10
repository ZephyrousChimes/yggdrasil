"""FastAPI serving layer for the trained retrieval + ranker pipeline.

Mirrors a real two-stage serving path: retrieval narrows the full catalog to
a shortlist via embedding lookup, the ranker rescoring only runs over that
shortlist -- never the full item catalog.
"""
import json
import pickle
import time
from pathlib import Path

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException

from retrieval import TwoTowerModel
from ranker import build_features

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
RETRIEVAL_K = 50
FINAL_K = 10

app = FastAPI(title="recsys-mvp")

with open(ARTIFACT_DIR / "meta.json") as f:
    META = json.load(f)
user2idx = {int(k): v for k, v in META["user2idx"].items()}
item2idx = {int(k): v for k, v in META["item2idx"].items()}
idx2item = {v: k for k, v in item2idx.items()}
feature_cols = META["feature_cols"]
fallback_recs = META["top_popular_fallback"]

model = TwoTowerModel(META["n_users"], META["n_items"], dim=META["dim"])
model.load_state_dict(torch.load(ARTIFACT_DIR / "retrieval_model.pt"))
model.eval()

with open(ARTIFACT_DIR / "ranker.pkl", "rb") as f:
    ranker = pickle.load(f)

items = pd.read_json(ARTIFACT_DIR / "items.json")
item_pop = pd.read_json(ARTIFACT_DIR / "item_popularity.json", typ="series")

with torch.no_grad():
    ITEM_EMBS = model.all_item_embeddings()


@app.get("/health")
def health():
    return {"status": "ok", "n_users": META["n_users"], "n_items": META["n_items"]}


@app.get("/metrics")
def metrics():
    """Offline eval metrics computed at training time -- what a real
    monitoring dashboard would expose for the model currently deployed."""
    return META["metrics"]


@app.get("/recommend")
def recommend(user_id: int, k: int = FINAL_K):
    start = time.perf_counter()

    if user_id not in user2idx:
        # cold-start: no embedding exists for a user we've never seen, so
        # fall back to popularity instead of returning garbage from an
        # untrained embedding row.
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "user_id": user_id,
            "recommendations": fallback_recs[:k],
            "strategy": "popularity_fallback_cold_start",
            "latency_ms": round(latency_ms, 2),
        }

    with torch.no_grad():
        u_emb = model.user_embedding(user2idx[user_id])
        scores = ITEM_EMBS @ u_emb
        top = torch.topk(scores, k=min(RETRIEVAL_K, META["n_items"]))
        cand_items = [idx2item[i.item()] for i in top.indices]
        retrieval_scores = {
            (user_id, idx2item[i.item()]): s.item()
            for i, s in zip(top.indices, top.values)
        }

    cand_df = pd.DataFrame({"user_id": user_id, "item_id": cand_items})
    feats, _ = build_features(cand_df, items, item_pop, retrieval_scores)
    rank_scores = ranker.predict_proba(feats[feature_cols])[:, 1]
    ranked = [it for _, it in sorted(zip(rank_scores, cand_items), reverse=True)]

    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "user_id": user_id,
        "recommendations": ranked[:k],
        "strategy": "two_tower_retrieval_plus_gbdt_rerank",
        "latency_ms": round(latency_ms, 2),
    }
