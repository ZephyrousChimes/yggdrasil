"""Load MovieLens-100k, build a time-based split, and item content features."""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ml-100k"

RATING_COLS = ["user_id", "item_id", "rating", "timestamp"]
ITEM_COLS = [
    "item_id", "title", "release_date", "video_release_date", "imdb_url",
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "FilmNoir", "Horror",
    "Musical", "Mystery", "Romance", "SciFi", "Thriller", "War", "Western",
]
GENRE_COLS = ITEM_COLS[5:]


def load_ratings() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "u.data", sep="\t", names=RATING_COLS)
    # implicit feedback: treat rating >= 4 as a positive interaction, mirrors
    # click/order-style implicit signals instead of using the star rating as
    # a regression target.
    df["label"] = (df["rating"] >= 4).astype(int)
    return df


def load_items() -> pd.DataFrame:
    return pd.read_csv(
        DATA_DIR / "u.item", sep="|", names=ITEM_COLS, encoding="latin-1"
    )


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Split by timestamp, not randomly.

    A random split lets a user's future interactions leak into training and
    overstates offline metrics; splitting on a global timestamp cutoff
    matches how the model will actually be evaluated in production (predict
    forward, not interpolate).
    """
    df = df.sort_values("timestamp")
    cutoff = df["timestamp"].quantile(1 - test_frac)
    train = df[df["timestamp"] <= cutoff].copy()
    test = df[df["timestamp"] > cutoff].copy()
    # drop test interactions for users/items never seen in training -
    # otherwise the retrieval model has no embedding to look up (cold-start
    # is handled separately, not silently by leaking future rows into train).
    test = test[
        test["user_id"].isin(train["user_id"]) & test["item_id"].isin(train["item_id"])
    ]
    return train, test


if __name__ == "__main__":
    ratings = load_ratings()
    train, test = time_based_split(ratings)
    print(f"ratings={len(ratings)} train={len(train)} test={len(test)}")
    print(f"positives in train={train['label'].sum()} test={test['label'].sum()}")
