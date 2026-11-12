import numpy as np
import torch


@torch.no_grad()
def topk(query, gallery, k=10, rotations=1, mask=None, chunk=1024):
    """Indices of the k best gallery locations per query. Embeddings must be L2-normalized.
    With rotations > 1 the gallery holds each location `rotations` times in a row
    (one per heading), and a location scores as its best heading.
    mask (Q, G) bool, optional: only these locations can be retrieved."""
    out = []
    for start in range(0, len(query), chunk):
        sim = query[start:start + chunk] @ gallery.T
        sim = sim.view(len(sim), -1, rotations).amax(dim=2)
        if mask is not None:
            sim = sim.masked_fill(~mask[start:start + chunk], float("-inf"))
        out.append(sim.topk(min(k, sim.shape[1]), dim=1).indices)
    return torch.cat(out).numpy()


def search_mask(query_xy, gallery_xy, search_m, seed=0):
    """Fact-check setting: the caption already names a place. Search only within search_m of a
    claimed point, which lies up to search_m / 2 from where the photo was really taken."""
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 2 * np.pi, len(query_xy))
    r = search_m / 2 * np.sqrt(rng.uniform(0, 1, len(query_xy)))
    claimed = query_xy + r[:, None] * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    return np.linalg.norm(gallery_xy[None] - claimed[:, None], axis=-1) <= search_m


def retrieval_metrics(top, query_xy, gallery_xy, radius_m=50, ks=(1, 5, 10), mask=None):
    """A retrieved tile is correct if its center is within radius_m of the photo."""
    dist = np.linalg.norm(gallery_xy[top] - query_xy[:, None], axis=-1)  # (Q, k)
    out = {f"R@{k}": 100 * np.mean(dist[:, :k].min(1) <= radius_m) for k in ks}
    out["median_m"] = np.median(dist[:, 0])

    # reference points: random ranking, and the closest tile the gallery has
    chance, random, oracle = [], [], []
    for idx in np.array_split(np.arange(len(query_xy)), max(1, len(query_xy) // 256)):
        d = np.linalg.norm(gallery_xy[None] - query_xy[idx, None], axis=-1)  # (chunk, G)
        allowed = np.ones_like(d, dtype=bool) if mask is None else mask[idx]
        chance.append(((d <= radius_m) & allowed).sum(1) / allowed.sum(1))
        d = np.where(allowed, d, np.nan)
        random.append(np.nanmedian(d, axis=1))
        oracle.append(np.nanmin(d, axis=1))
    out["chance_R@1"] = 100 * np.mean(np.concatenate(chance))
    out["median_m_random"] = np.median(np.concatenate(random))
    out["median_m_oracle"] = np.median(np.concatenate(oracle))
    return {k: round(float(v), 2) for k, v in out.items()}
