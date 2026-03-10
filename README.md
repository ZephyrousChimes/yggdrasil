# recsys-mvp

Two-stage recommender (retrieval + ranking) on MovieLens-100k, served over FastAPI.

## Architecture

```
user_id --> [two-tower retrieval] --> top-50 candidates --> [GBDT ranker] --> top-10
              (embedding dot product,                          (genres +
               scans full catalog cheaply)                      popularity +
                                                                  retrieval score)
```

Two stages instead of one model: retrieval has to score the entire catalog
per request cheaply (embedding lookup + dot product), so it can only afford
user/item embeddings as features. The ranker only ever scores the ~50
candidates retrieval already shortlisted, so it can afford richer,
more expensive features (content, popularity) that wouldn't scale to the
full catalog.

## Key design decisions

- **Implicit labels, not raw ratings**: rating >= 4 is treated as a positive
  interaction, matching how real click/order signals work, instead of
  training a regressor on the 1-5 star scale.
- **In-batch negative sampling for retrieval**: negatives come from other
  items in the same training batch rather than uniform-random sampling.
  Random negatives are almost always trivially distinguishable and teach a
  decision boundary that doesn't transfer to the hard negatives the model
  actually sees at serving time.
- **Time-based train/test split**: split on a global timestamp cutoff, not
  a random split. A random split lets a user's future interactions leak
  into training and overstates offline metrics; a time split matches how
  the model is actually evaluated in production (predict forward).
- **Cold-start fallback**: a user_id never seen in training has no trained
  embedding row, so `/recommend` falls back to a popularity list instead of
  returning a score from an untrained embedding.
- **Evaluation metric**: Recall@K / NDCG@K, not accuracy. This is a ranking
  task where *position* in the returned list matters, not just presence in
  the candidate pool.

## Results (offline, held-out time-split test set, k=10)

| Strategy                     | Recall@10 | NDCG@10 |
|-------------------------------|-----------|---------|
| Popularity baseline            | 0.035     | 0.075   |
| Retrieval only (no rerank)     | 0.015     | 0.021   |
| Retrieval + GBDT rerank        | 0.024     | 0.046   |

**Honest read of these numbers**: the popularity baseline beats the learned
retrieval model here. This is a known, documented effect in recsys
literature on small/sparse implicit-feedback datasets (MovieLens-100k is
~100K interactions across 943 users / 1682 items) -- popularity is a
surprisingly strong baseline because it directly exploits the same
skew the evaluation set inherits. Adding popularity as a ranker feature
closes about a third of the gap. Next steps to actually beat it: hard
negative mining (instead of in-batch), deeper towers with side features
(genres, recency), and more training epochs on a larger interaction log.
This is left as-is deliberately rather than tuned to "win" against the
baseline, since the point of the eval harness is to surface this kind of
result, not hide it.

## Run it

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash data/download.sh          # fetches MovieLens-100k
python src/train.py            # trains retrieval + ranker, writes artifacts/
uvicorn serve:app --app-dir src --port 8001
curl "http://127.0.0.1:8001/recommend?user_id=1&k=5"
curl "http://127.0.0.1:8001/metrics"
```

## Files

- `src/data.py` -- loading, implicit-label construction, time-based split
- `src/retrieval.py` -- two-tower model + in-batch-negative training loop
- `src/ranker.py` -- feature construction + GBDT reranker
- `src/eval.py` -- Recall@K / NDCG@K
- `src/train.py` -- end-to-end pipeline, writes `artifacts/`
- `src/serve.py` -- FastAPI serving layer with cold-start fallback and a
  `/metrics` endpoint exposing the offline eval numbers for the deployed model
