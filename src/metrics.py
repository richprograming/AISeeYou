import numpy as np
import torch


@torch.no_grad()
def topk(query, gallery, k=10, rotations=1, chunk=1024):
    """Indices of the k best gallery locations per query. Embeddings must be L2-normalized.
    With rotations > 1 the gallery holds each location `rotations` times in a row
    (one per heading), and a location scores as its best heading."""
    out = []
    for start in range(0, len(query), chunk):
        sim = query[start:start + chunk] @ gallery.T
        sim = sim.view(len(sim), -1, rotations).amax(dim=2)
        out.append(sim.topk(min(k, sim.shape[1]), dim=1).indices)
    return torch.cat(out).numpy()


def retrieval_metrics(top, query_xy, gallery_xy, radius_m=50, ks=(1, 5, 10)):
    """A retrieved tile is correct if its center is within radius_m of the photo."""
    dist = np.linalg.norm(gallery_xy[top] - query_xy[:, None], axis=-1)  # (Q, k)
    out = {f"R@{k}": 100 * np.mean(dist[:, :k].min(1) <= radius_m) for k in ks}
    out["median_m"] = np.median(dist[:, 0])

    # reference points: random ranking, and the closest tile the gallery has
    chance, random, oracle = [], [], []
    for q in np.array_split(query_xy, max(1, len(query_xy) // 256)):
        d = np.linalg.norm(gallery_xy[None] - q[:, None], axis=-1)  # (chunk, G)
        chance.append(np.mean(d <= radius_m, axis=1))
        random.append(np.median(d, axis=1))
        oracle.append(d.min(1))
    out["chance_R@1"] = 100 * np.mean(np.concatenate(chance))
    out["median_m_random"] = np.median(np.concatenate(random))
    out["median_m_oracle"] = np.median(np.concatenate(oracle))
    return {k: round(float(v), 2) for k, v in out.items()}
