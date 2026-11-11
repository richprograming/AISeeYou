import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from src.geo import to_xy


@dataclass
class City:
    name: str
    photos: list         # photo paths
    xy: np.ndarray       # (N, 2) photo positions in meters, origin at the map center
    heading: np.ndarray  # (N,) camera heading, degrees clockwise from north (nan if unknown)
    is_pano: np.ndarray  # (N,) bool
    raster: np.ndarray   # (H, W, 3) rendered OSM map
    size_m: float
    m_per_px: float


def to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


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
        heading=np.array([to_float(r["compass"]) for r in rows]),
        is_pano=np.array([r["is_pano"] == "True" for r in rows], dtype=bool),
        raster=raster,
        size_m=meta["size_m"],
        m_per_px=meta["m_per_px"],
    )


def select(city, idx):
    idx = np.asarray(idx, dtype=int)
    return City(city.name, [city.photos[i] for i in idx], city.xy[idx], city.heading[idx],
                city.is_pano[idx], city.raster, city.size_m, city.m_per_px)


def grid(size_m, tile_m, stride_m):
    """Centers of gallery tiles covering the map, keeping every tile fully inside it."""
    half = size_m / 2 - tile_m / 2
    ticks = np.arange(-half, half + 1e-6, stride_m)
    xs, ys = np.meshgrid(ticks, ticks)
    return np.stack([xs.ravel(), ys.ravel()], axis=1)


def crop(raster, x, y, tile_m, m_per_px, size_m, heading=0.0):
    """Square map tile of side tile_m centered at (x, y) meters, rotated so that
    `heading` (degrees clockwise from north) points up. heading=0 gives north up."""
    side = tile_m * 1.5 if heading else tile_m  # margin so rotated corners stay filled
    half = side / 2 / m_per_px
    col = (x + size_m / 2) / m_per_px
    row = (size_m / 2 - y) / m_per_px
    left, top, right, bottom = (round(v) for v in (col - half, row - half, col + half, row + half))
    if left >= 0 and top >= 0 and right <= raster.shape[1] and bottom <= raster.shape[0]:
        tile = Image.fromarray(raster[top:bottom, left:right])
    else:
        tile = Image.fromarray(raster).crop((left, top, right, bottom))  # pads outside the map
    if not heading:
        return tile
    # PIL rotates counterclockwise, which brings a feature at `heading` clockwise to the top
    tile = tile.rotate(heading, resample=Image.NEAREST)
    keep = round(tile_m / m_per_px)
    off = (tile.width - keep) // 2
    return tile.crop((off, off, off + keep, off + keep))


class Photos(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB"))


def pano_view(img, aspect):
    """Perspective-like view from an equirectangular panorama: middle band, random direction.
    Also returns the view direction relative to the panorama center, in degrees."""
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    band = arr[h // 4: 3 * h // 4]
    width = int(band.shape[0] * aspect)
    shift = random.randrange(w)
    view = np.roll(band, -shift, axis=1)[:, :width]
    offset = ((shift + width / 2) / w - 0.5) * 360
    return Image.fromarray(view), offset


class Pairs(Dataset):
    """Training pairs: a photo and the map tile around where it was taken.
    With align=True the tile is rotated so the camera direction points up; otherwise north up.
    The tile center is shifted by up to jitter_m so the model does not rely on exact centering."""

    def __init__(self, cities, tile_m, photo_transform, tile_transform, jitter_m=0, align=False,
                 jitter_deg=10, aspect=1.375):
        self.cities = cities
        self.items = [(c, i) for c, city in enumerate(cities) for i in range(len(city.photos))]
        self.tile_m = tile_m
        self.photo_transform = photo_transform
        self.tile_transform = tile_transform
        self.jitter_m = jitter_m
        self.align = align
        self.jitter_deg = jitter_deg
        self.aspect = aspect

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        c, i = self.items[k]
        city = self.cities[c]
        photo = Image.open(city.photos[i]).convert("RGB")
        heading = city.heading[i]
        if city.is_pano[i]:
            photo, offset = pano_view(photo, self.aspect)
            heading += offset

        heading = (heading + random.uniform(-self.jitter_deg, self.jitter_deg)) % 360 if self.align else 0.0
        limit = city.size_m / 2 - self.tile_m / 2
        x, y = (np.clip(v + random.uniform(-self.jitter_m, self.jitter_m), -limit, limit) for v in city.xy[i])
        tile = crop(city.raster, x, y, self.tile_m, city.m_per_px, city.size_m, heading)
        xy = torch.tensor(city.xy[i], dtype=torch.float32)
        return self.photo_transform(photo), self.tile_transform(tile), xy, c


class NearbyBatches(Sampler):
    """Batches built from groups of photos taken near each other (min_m to near_m apart, same city).
    In-batch negatives then look alike, which makes them hard. group=1 gives random batches."""

    def __init__(self, cities, batch_size, group, near_m=400, min_m=50, seed=0):
        self.xy = [c.xy for c in cities]
        self.offsets = np.cumsum([0] + [len(c.photos) for c in cities])
        self.batch_size, self.group = batch_size, group
        self.near_m, self.min_m = near_m, min_m
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.offsets[-1] // self.batch_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        used = np.zeros(self.offsets[-1], dtype=bool)
        batch = []
        for anchor in rng.permutation(self.offsets[-1]):
            if used[anchor]:
                continue
            c = np.searchsorted(self.offsets, anchor, side="right") - 1
            start, xy = self.offsets[c], self.xy[c]
            dist = np.linalg.norm(xy - xy[anchor - start], axis=1)
            near = np.flatnonzero((dist > self.min_m) & (dist < self.near_m) & ~used[start:start + len(xy)])
            picked = rng.choice(near, min(self.group - 1, len(near)), replace=False) + start
            members = [anchor, *picked]
            used[members] = True
            batch += members
            while len(batch) >= self.batch_size:
                yield [int(i) for i in batch[:self.batch_size]]
                batch = batch[self.batch_size:]


class Tiles(Dataset):
    """Gallery tiles: every center, at each of `rotations` evenly spaced headings
    (item i is center i // rotations at heading (i % rotations) * 360 / rotations)."""

    def __init__(self, city, centers, tile_m, transform, rotations=1):
        self.city = city
        self.centers = centers
        self.tile_m = tile_m
        self.transform = transform
        self.rotations = rotations

    def __len__(self):
        return len(self.centers) * self.rotations

    def __getitem__(self, i):
        x, y = self.centers[i // self.rotations]
        heading = (i % self.rotations) * 360 / self.rotations
        c = self.city
        return self.transform(crop(c.raster, x, y, self.tile_m, c.m_per_px, c.size_m, heading))
