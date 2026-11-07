import numpy as np
import torch
import torch.nn.functional as F

from src.metrics import retrieval_metrics, topk


def test_topk_order():
    query = torch.tensor([[1.0, 0.0]])
    gallery = F.normalize(torch.tensor([[0.0, 1.0], [1.0, 0.1], [1.0, 0.0]]), dim=-1)
    assert topk(query, gallery, k=3)[0].tolist() == [2, 1, 0]


def test_perfect_and_wrong_retrieval():
    gallery_xy = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
    query_xy = np.array([[10.0, 0.0], [190.0, 0.0]])
    good = retrieval_metrics(np.array([[0, 1], [2, 1]]), query_xy, gallery_xy, ks=(1, 2))
    bad = retrieval_metrics(np.array([[2, 1], [0, 1]]), query_xy, gallery_xy, ks=(1, 2))
    assert good["R@1"] == 100 and good["median_m"] == 10
    assert bad["R@1"] == 0 and bad["R@2"] == 0 and bad["median_m"] == 190
    assert good["median_m_oracle"] == 10
    assert abs(good["chance_R@1"] - 100 / 3) < 0.01
