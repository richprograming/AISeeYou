import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from src.geo import to_xy


@dataclass
class City:
    name: str
    photos: list         # photo paths
    xy: np.ndarray       # (N, 2) photo positions in meters, origin at the map center
    is_pano: np.ndarray  # (N,) bool
    raster: np.ndarray   # (H, W, 3) rendered OSM map
    size_m: float
    m_per_px: float


def load_city(root, name):
    d = Path(root) / name
    meta = json.loads((d / "map.json").read_text())
    raster = np.asarray(Image.open(d / "map.png").convert("RGB"))
    with open(d / "photos.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    lat = [float(r["lat"]) for r in rows]
    lon = [float(r["lon"]) for r in rows]
    return City(
        name=name,
        photos=[d / "photos" / f"{r['id']}.jpg" for r in rows],
        xy=to_xy(lat, lon, meta["lat"], meta["lon"]).reshape(-1, 2),
        is_pano=np.array([r["is_pano"] == "True" for r in rows], dtype=bool),
        raster=raster,
        size_m=meta["size_m"],
        m_per_px=meta["m_per_px"],
    )


def select(city, idx):
    idx = np.asarray(idx)
    return City(city.name, [city.photos[i] for i in idx], city.xy[idx], city.is_pano[idx],
                city.raster, city.size_m, city.m_per_px)


def grid(size_m, tile_m, stride_m):
    """Centers of gallery tiles covering the map, keeping every tile fully inside it."""
    half = size_m / 2 - tile_m / 2
    ticks = np.arange(-half, half + 1e-6, stride_m)
    xs, ys = np.meshgrid(ticks, ticks)
    return np.stack([xs.ravel(), ys.ravel()], axis=1)


def crop(raster, x, y, tile_m, m_per_px, size_m):
    """Square map tile of side tile_m centered at (x, y) meters, north up."""
    half = tile_m / 2 / m_per_px
    col = (x + size_m / 2) / m_per_px
    row = (size_m / 2 - y) / m_per_px
    box = tuple(round(v) for v in (col - half, row - half, col + half, row + half))
    return Image.fromarray(raster).crop(box)


class Photos(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB"))


class Tiles(Dataset):
    def __init__(self, city, centers, tile_m, transform):
        self.city = city
        self.centers = centers
        self.tile_m = tile_m
        self.transform = transform

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, i):
        x, y = self.centers[i]
        c = self.city
        return self.transform(crop(c.raster, x, y, self.tile_m, c.m_per_px, c.size_m))
