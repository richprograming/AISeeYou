import csv
import json

import numpy as np
import pytest
from PIL import Image

from src.data import crop, grid
from src.geo import to_latlon

LAT0, LON0 = -30.0277, -51.2287
SIZE_M, M_PER_PX, TILE_M = 1000, 1.0, 100


@pytest.fixture
def fake_city(tmp_path):
    return make_fake_city(tmp_path), "fake"


def make_fake_city(root):
    """City whose 'photos' are map crops taken exactly at gallery tile centers."""
    rng = np.random.default_rng(0)
    d = root / "fake"
    (d / "photos").mkdir(parents=True)
    small = rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)
    raster = np.asarray(Image.fromarray(small).resize((1000, 1000), Image.NEAREST))
    Image.fromarray(raster).save(d / "map.png")
    meta = {"lat": LAT0, "lon": LON0, "size_m": SIZE_M, "m_per_px": M_PER_PX}
    (d / "map.json").write_text(json.dumps(meta))

    centers = grid(SIZE_M, TILE_M, 50)
    picks = rng.choice(len(centers), 20, replace=False)
    with open(d / "photos.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "lat", "lon", "compass", "is_pano", "captured_at", "sequence"])
        for i, (x, y) in enumerate(centers[picks]):
            crop(raster, x, y, TILE_M, M_PER_PX, SIZE_M).save(d / "photos" / f"{i}.jpg")
            lat, lon = to_latlon(x, y, LAT0, LON0)
            w.writerow([i, lat, lon, 0.0, i % 2 == 0, 0, "seq"])
    return root
