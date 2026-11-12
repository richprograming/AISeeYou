import numpy as np
import torch
import torch.nn.functional as F

from src.metrics import retrieval_metrics, search_mask, topk


def test_topk_order():
    query = torch.tensor([[1.0, 0.0]])
    gallery = F.normalize(torch.tensor([[0.0, 1.0], [1.0, 0.1], [1.0, 0.0]]), dim=-1)
    assert topk(query, gallery, k=3)[0].tolist() == [2, 1, 0]


def test_topk_scores_location_by_best_rotation():
    query = torch.tensor([[1.0, 0.0]])
    # 2 locations x 2 rotations: location 1 matches only at its second rotation
    gallery = torch.tensor([[0.6, 0.8], [0.6, 0.8], [0.0, 1.0], [1.0, 0.0]])
    assert topk(query, gallery, k=2, rotations=2)[0].tolist() == [1, 0]


def test_perfect_and_wrong_retrieval():
    gallery_xy = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
    query_xy = np.array([[10.0, 0.0], [190.0, 0.0]])
    good = retrieval_metrics(np.array([[0, 1], [2, 1]]), query_xy, gallery_xy, ks=(1, 2))
    bad = retrieval_metrics(np.array([[2, 1], [0, 1]]), query_xy, gallery_xy, ks=(1, 2))
    assert good["R@1"] == 100 and good["median_m"] == 10
    assert bad["R@1"] == 0 and bad["R@2"] == 0 and bad["median_m"] == 190
    assert good["median_m_oracle"] == 10
    assert abs(good["chance_R@1"] - 100 / 3) < 0.01


def test_search_mask_always_contains_the_photo():
    query_xy = np.random.default_rng(1).uniform(-500, 500, (200, 2))
    mask = search_mask(query_xy, query_xy, search_m=300)
    assert mask.diagonal().all()
    assert not mask.all()


def test_mask_limits_retrieval_and_chance():
    query = torch.tensor([[1.0, 0.0]])
    gallery = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    mask = np.array([[False, True, True]])
    top = topk(query, gallery, k=2, mask=torch.from_numpy(mask))
    assert top[0].tolist() == [1, 2]
    gallery_xy = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
    m = retrieval_metrics(top, np.array([[100.0, 0.0]]), gallery_xy, ks=(1,), mask=mask)
    assert m["R@1"] == 100 and m["chance_R@1"] == 50
