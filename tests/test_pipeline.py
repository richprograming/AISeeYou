import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from scripts.make_fake_vigor import make_fake_vigor
from src.data import load_vigor, subset
from src.metrics import rank_queries, retrieval_metrics


def thumbnail(path):
    x = np.asarray(Image.open(path).convert("RGB").resize((8, 8), Image.BILINEAR), dtype=np.float32).ravel()
    return x - x.mean()


def test_labels_end_to_end(tmp_path):
    # pixel thumbnails match each fake pano to its tile, so anything below 100% is a wiring bug
    data = load_vigor(make_fake_vigor(tmp_path, n=36), ["Chicago"], "same_area_balanced_test.txt")
    data = subset(data, 12, seed=3)
    q = F.normalize(torch.tensor(np.stack([thumbnail(p) for p in data.panos])), dim=-1)
    g = F.normalize(torch.tensor(np.stack([thumbnail(p) for p in data.sats])), dim=-1)
    ranks, top1 = rank_queries(q, g, data.labels[:, 0])
    m = retrieval_metrics(ranks, top1, data.labels, data.pano_coords(), data.sat_coords())
    assert m["R@1"] == 100
    assert m["median_m"] == m["median_m_oracle"]
