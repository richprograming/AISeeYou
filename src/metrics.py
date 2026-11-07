import numpy as np
import torch

EARTH_RADIUS_M = 6_371_000


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


@torch.no_grad()
def rank_queries(query, gallery, positive, chunk=1024):
    """Rank of the positive for each query (0 = top-1) and the top-1 gallery index.
    Embeddings must be L2-normalized. Chunked so the full similarity matrix never
    has to fit in memory."""
    ranks, top1 = [], []
    for start in range(0, len(query), chunk):
        sim = query[start:start + chunk] @ gallery.T
        pos = torch.as_tensor(positive[start:start + chunk])
        pos_score = sim[torch.arange(len(sim)), pos]
        ranks.append((sim > pos_score[:, None]).sum(1))
        top1.append(sim.argmax(1))
    return torch.cat(ranks).numpy(), torch.cat(top1).numpy()


def retrieval_metrics(ranks, top1, labels, query_coords, gallery_coords, ks=(1, 5, 10)):
    out = {f"R@{k}": 100 * np.mean(ranks < k) for k in ks}
    one_percent = max(1, round(0.01 * len(gallery_coords)))
    out["R@1%"] = 100 * np.mean(ranks < one_percent)
    # VIGOR hit rate: top-1 is the positive or one of the semi-positives
    out["hit"] = 100 * np.mean((labels == top1[:, None]).any(1))

    q, g = query_coords, gallery_coords
    err = haversine(q[:, 0], q[:, 1], g[top1, 0], g[top1, 1])
    out["median_m"] = float(np.median(err))
    # lower bound: even a perfect top-1 is off by the pano-to-tile-center offset
    oracle = haversine(q[:, 0], q[:, 1], g[labels[:, 0], 0], g[labels[:, 0], 1])
    out["median_m_oracle"] = float(np.median(oracle))
    return {k: round(float(v), 2) for k, v in out.items()}
