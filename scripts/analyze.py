"""Error analysis from the embeddings saved by scripts.test.

python -m scripts.analyze outputs/test_latam [--root data/cities]

Per city, writes <dir>/analysis.json and <dir>/<city>_examples.jpg
(rows: photo | true tile, camera heading up | top-1 tile at its best rotation).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.data import crop, load_city
from src.osm import COLORS

RADIUS_M = 50
TILE_M = 100


def first_correct_rank(sim, dist):
    order = np.argsort(-sim, axis=1)
    return (np.take_along_axis(dist, order, 1) <= RADIUS_M).argmax(1), order[:, 0]


def empty_share(city, xy):
    """Fraction of the tile around each photo where OSM has nothing drawn."""
    bg = np.array(COLORS["background"])
    return np.array([np.mean((np.asarray(crop(city.raster, x, y, TILE_M, city.m_per_px, city.size_m)) == bg).all(-1))
                     for x, y in xy])


def examples(city, z, rows, top1, rot_deg, dist_top1, path, size=256):
    sheet = Image.new("RGB", (size * 3, size * len(rows)), "white")
    for r, i in enumerate(rows):
        photo = Image.open(Path(city_dir(city)) / "photos" / str(z["photos"][i])).convert("RGB").resize((size, size))
        x, y = z["query_xy"][i]
        true = crop(city.raster, x, y, TILE_M, city.m_per_px, city.size_m, z["heading"][i] or 360)
        cx, cy = z["centers"][top1[i]]
        pred = crop(city.raster, cx, cy, TILE_M, city.m_per_px, city.size_m, rot_deg[i] or 360)
        for c, img in enumerate([photo, true.resize((size, size)), pred.resize((size, size))]):
            sheet.paste(img, (c * size, r * size))
        ImageDraw.Draw(sheet).text((2 * size + 4, r * size + 4), f"{dist_top1[i]:.0f} m away", fill="black")
    sheet.save(path, quality=90)


def city_dir(city):
    return Path(city.photos[0]).parent.parent if len(city.photos) else "."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("--root", default="data/cities")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = {}
    for path in sorted(Path(args.dir).glob("*.npz")):
        z = np.load(path)
        city = load_city(args.root, path.stem)
        q = torch.tensor(z["query"].astype(np.float32))
        g = torch.tensor(z["gallery"].astype(np.float32))
        n = len(z["centers"])
        sim = (q @ g.T).view(len(q), n, -1)
        rotations = sim.shape[2]
        best_rot = sim.argmax(2).numpy()
        sim = sim.amax(2).numpy()
        dist = np.linalg.norm(z["centers"][None] - z["query_xy"][:, None], axis=-1)

        rank, top1 = first_correct_rank(sim, dist)
        random_rank, _ = first_correct_rank(rng.random(sim.shape), dist)
        rows = np.arange(len(q))
        dist_top1 = dist[rows, top1]

        # rotation the model prefers at the tile closest to the photo, against the compass
        step = 360 / rotations
        rot_true = best_rot[rows, dist.argmin(1)] * step
        rot_err = np.abs((rot_true - z["heading"] + 180) % 360 - 180)

        empty = empty_share(city, z["query_xy"])
        sparse = empty > np.median(empty)
        top = lambda r, pct, keep=slice(None): round(100 * float(np.mean(r[keep] < n * pct / 100)), 1)
        out[path.stem] = {
            "median_rank_of_first_correct": float(np.median(rank)), "gallery": n,
            "correct_in_top_%": {p: {"model": top(rank, p), "random": top(random_rank, p)} for p in (1, 5, 10)},
            "top1_within_m": {m: round(100 * float(np.mean(dist_top1 <= m)), 1) for m in (50, 100, 200)},
            "rotation_matches_compass_%": {"model": round(100 * float(np.mean(rot_err <= step / 2)), 1),
                                           "random": round(100 / rotations, 1)},
            "empty_map_share": round(float(empty.mean()), 2),
            "correct_in_top_5%_by_map": {"well_mapped": top(rank, 5, ~sparse), "sparse": top(rank, 5, sparse)},
        }
        print(path.stem, json.dumps(out[path.stem], indent=1))

        hits = rng.permutation(np.flatnonzero(dist_top1 <= RADIUS_M))[:3]
        misses = rng.permutation(np.flatnonzero(dist_top1 > 200))[:5]
        examples(city, z, [*hits, *misses], top1, best_rot[rows, top1] * step, dist_top1,
                 Path(args.dir) / f"{path.stem}_examples.jpg")

    json.dump(out, open(Path(args.dir) / "analysis.json", "w"), indent=2)


if __name__ == "__main__":
    main()
