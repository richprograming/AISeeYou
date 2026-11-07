import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data import crop, grid, load_city
from src.metrics import retrieval_metrics, topk


def test_grid_stays_inside_map():
    centers = grid(1000, 100, 25)
    assert len(centers) == 37 * 37
    assert np.abs(centers).max() == 450


def test_crop_is_north_up():
    raster = np.zeros((100, 100, 3), dtype=np.uint8)
    raster[10, 50] = 255  # 40 m north of the center at 1 m/px
    tile = np.asarray(crop(raster, 0, 40, 10, 1.0, 100))
    assert tile.shape == (10, 10, 3) and tile[5, 5, 0] == 255


def thumbnail(img):
    x = np.asarray(img.convert("RGB").resize((8, 8), Image.BILINEAR), dtype=np.float32).ravel()
    return x - x.mean()


def test_photos_match_their_tiles(fake_city):
    # pixel features find the exact tile, so anything short of 100% is a coordinate bug
    root, name = fake_city
    city = load_city(root, name)
    assert len(city.photos) == 20 and city.is_pano.sum() == 10
    centers = grid(city.size_m, 100, 50)
    q = torch.tensor(np.stack([thumbnail(Image.open(p)) for p in city.photos]))
    g = torch.tensor(np.stack([thumbnail(crop(city.raster, x, y, 100, city.m_per_px, city.size_m))
                               for x, y in centers]))
    top = topk(F.normalize(q, dim=-1), F.normalize(g, dim=-1))
    m = retrieval_metrics(top, city.xy, centers)
    assert m["R@1"] == 100 and m["median_m"] < 0.01
