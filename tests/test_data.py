import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data import NearbyBatches, crop, grid, load_city, pano_view
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


def test_pano_view_shape_and_direction():
    pano = np.zeros((512, 1024, 3), dtype=np.uint8)
    pano[:, 512] = 255  # the panorama center column (heading offset 0)
    for _ in range(20):
        view, offset = pano_view(Image.fromarray(pano), aspect=1.5)
        assert view.size == (384, 256)
        cols = np.flatnonzero(np.asarray(view)[0, :, 0] == 255)
        if len(cols):
            # the center column sits at -offset degrees from the view center
            expected = 192 - offset / 360 * 1024
            assert abs(cols[0] - expected) <= 1


class FakeCity:
    def __init__(self, xy):
        self.xy = np.asarray(xy, dtype=float)
        self.photos = list(range(len(xy)))


def test_nearby_batches_use_every_photo_once():
    rng = np.random.default_rng(0)
    cities = [FakeCity(rng.uniform(-900, 900, (101, 2))), FakeCity(rng.uniform(-900, 900, (60, 2)))]
    sampler = NearbyBatches(cities, batch_size=16, group=4, near_m=400, min_m=50)
    batches = list(sampler)
    flat = [i for b in batches for i in b]
    assert len(batches) == len(sampler) == 161 // 16
    assert len(flat) == len(set(flat))


def test_nearby_batches_groups_are_close():
    xy = [[0, 0], [100, 0], [200, 0], [5000, 0], [5100, 0], [5200, 0]]
    sampler = NearbyBatches([FakeCity(xy)], batch_size=3, group=3, near_m=400, min_m=50)
    for batch in sampler:
        assert len({i // 3 for i in batch}) == 1  # all from the same cluster


def test_crop_rotates_heading_to_top():
    raster = np.zeros((200, 200, 3), dtype=np.uint8)
    raster[100, 130] = 255  # 30 m east of the center at 1 m/px
    tile = np.asarray(crop(raster, 0, 0, 100, 1.0, 200, heading=90))
    assert tile.shape == (100, 100, 3)
    row, col = np.argwhere(tile[..., 0] == 255)[0]
    assert abs(row - 20) <= 1 and abs(col - 50) <= 1  # now 30 px above the center


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
