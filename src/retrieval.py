"""Two-tower embedding model for candidate retrieval, trained with
in-batch negatives.

In-batch negatives (rather than uniformly-random negatives) are used because
they're drawn from the same distribution the model will actually be scored
against at serving time -- random negatives are almost always trivially easy
and teach the model a decision boundary that doesn't transfer to the hard
negatives it will see in production.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class TwoTowerModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)

    def forward(self, user_ids, item_ids):
        u = F.normalize(self.user_emb(user_ids), dim=-1)
        v = F.normalize(self.item_emb(item_ids), dim=-1)
        return u, v

    def all_item_embeddings(self):
        return F.normalize(self.item_emb.weight, dim=-1)

    def user_embedding(self, user_id: int):
        return F.normalize(self.user_emb.weight[user_id], dim=-1)


class InteractionDataset(Dataset):
    def __init__(self, user_idx, item_idx):
        self.user_idx = user_idx
        self.item_idx = item_idx

    def __len__(self):
        return len(self.user_idx)

    def __getitem__(self, i):
        return self.user_idx[i], self.item_idx[i]


def in_batch_softmax_loss(u, v, temperature: float = 0.1):
    """Each row's positive is its diagonal match; every other item in the
    batch acts as a free negative -- this is the in-batch negative trick."""
    logits = (u @ v.T) / temperature
    labels = torch.arange(u.size(0), device=u.device)
    return F.cross_entropy(logits, labels)


def train_retrieval(
    user_idx, item_idx, n_users, n_items, dim=32, epochs=8, batch_size=512, lr=0.01
):
    model = TwoTowerModel(n_users, n_items, dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = InteractionDataset(user_idx, item_idx)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for u_ids, i_ids in dl:
            opt.zero_grad()
            u, v = model(u_ids, i_ids)
            loss = in_batch_softmax_loss(u, v)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"[retrieval] epoch {epoch+1}/{epochs} loss={total_loss/len(dl):.4f}")
    return model
