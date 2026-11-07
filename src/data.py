from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def pano_latlon(path):
    # "<panoid>,<lat>,<lon>,.jpg"
    parts = Path(path).name.split(",")
    return float(parts[-3]), float(parts[-2])


def sat_latlon(path):
    # "satellite_<lat>_<lon>.png"
    _, lat, lon = Path(path).stem.split("_")
    return float(lat), float(lon)


@dataclass
class Vigor:
    panos: list    # query image paths
    sats: list     # gallery image paths
    labels: np.ndarray  # (num_panos, 4) gallery indices: positive, then 3 semi-positives

    def pano_coords(self):
        return np.array([pano_latlon(p) for p in self.panos])

    def sat_coords(self):
        return np.array([sat_latlon(s) for s in self.sats])


def load_vigor(root, cities, split_file, label_dir="splits"):
    root = Path(root)
    panos, sats, labels = [], [], []
    for city in cities:
        names = (root / label_dir / city / "satellite_list.txt").read_text().split()
        index = {name: len(sats) + i for i, name in enumerate(names)}
        sats += [root / city / "satellite" / n for n in names]

        # line: pano sat_pos dx dy sat_semi1 dx dy sat_semi2 dx dy sat_semi3 dx dy
        for line in (root / label_dir / city / split_file).read_text().splitlines():
            tokens = line.split()
            if not tokens:
                continue
            panos.append(root / city / "panorama" / tokens[0])
            labels.append([index[tokens[i]] for i in (1, 4, 7, 10)])
    return Vigor(panos, sats, np.array(labels, dtype=np.int64))


def subset(data, num_queries, seed=0, full_gallery=False):
    """Keep num_queries random panoramas. Unless full_gallery, the gallery is
    reduced to the satellites those panoramas reference (much easier task)."""
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(data.panos), size=min(num_queries, len(data.panos)), replace=False)
    keep.sort()
    panos = [data.panos[i] for i in keep]
    labels = data.labels[keep]
    if full_gallery:
        return Vigor(panos, data.sats, labels)

    used = np.unique(labels)
    remap = {old: new for new, old in enumerate(used)}
    sats = [data.sats[i] for i in used]
    labels = np.vectorize(remap.get)(labels)
    return Vigor(panos, sats, labels)


class ImageList(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img)
