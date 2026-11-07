import numpy as np
import torch
import torch.nn.functional as F

from src.metrics import haversine, rank_queries, retrieval_metrics


def test_haversine_one_degree_latitude():
    assert abs(haversine(0.0, 0.0, 1.0, 0.0) - 111_195) < 10


def test_perfect_retrieval():
    emb = F.normalize(torch.randn(20, 8), dim=-1)
    labels = np.stack([np.arange(20)] * 4, axis=1)
    ranks, top1 = rank_queries(emb, emb, labels[:, 0], chunk=7)
    coords = np.random.rand(20, 2)
    m = retrieval_metrics(ranks, top1, labels, coords, coords)
    assert m["R@1"] == 100 and m["hit"] == 100 and m["median_m"] == 0


def test_rank_counts_better_items():
    query = torch.tensor([[1.0, 0.0]])
    gallery = F.normalize(torch.tensor([[0.0, 1.0], [1.0, 0.1], [1.0, 0.0]]), dim=-1)
    ranks, top1 = rank_queries(query, gallery, np.array([1]))
    assert ranks[0] == 1 and top1[0] == 2
