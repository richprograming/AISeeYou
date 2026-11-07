"""Download street-level photos from Mapillary inside each city area.

Needs a free client token: set MAPILLARY_TOKEN in the environment.

python -m scripts.fetch_mapillary sao_paulo --count-only     # coverage check, no images
python -m scripts.fetch_mapillary sao_paulo --max-photos 3000

Writes data/cities/<city>/photos.csv and photos/<id>.jpg (1024 px wide thumbnails).
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import requests
import yaml

from src.geo import bbox, to_xy

API_URL = "https://graph.mapillary.com/images"
FIELDS = "id,computed_geometry,geometry,compass_angle,captured_at,is_pano,sequence,thumb_1024_url"
CELL_DEG = 0.005    # query in small cells; the API caps results per request
MARGIN_M = 50       # keep photos away from the map border so tiles around them exist
THIN_M = 10         # at most one photo per 10 m cell, to avoid near duplicates


def list_images(token, s, w, n, e):
    images = []
    for lat in np.arange(s, n, CELL_DEG):
        for lon in np.arange(w, e, CELL_DEG):
            cell = f"{lon},{lat},{min(lon + CELL_DEG, e)},{min(lat + CELL_DEG, n)}"
            params = {"access_token": token, "fields": FIELDS, "bbox": cell, "limit": 2000}
            resp = requests.get(API_URL, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()["data"]
            if len(data) == params["limit"]:
                print(f"  cell {cell} hit the result cap, coverage there is partial")
            images += data
    return images


def thin(images, lat0, lon0, max_photos, seed=0):
    # newest first, so photos match the current OSM map as closely as possible
    images = sorted(images, key=lambda im: im.get("captured_at", 0), reverse=True)
    kept, seen = [], set()
    for im in images:
        lon, lat = (im.get("computed_geometry") or im["geometry"])["coordinates"]
        x, y = to_xy(lat, lon, lat0, lon0)
        cell = (int(x // THIN_M), int(y // THIN_M))
        if cell in seen or "thumb_1024_url" not in im:
            continue
        seen.add(cell)
        kept.append({**im, "lat": lat, "lon": lon})
    if max_photos and len(kept) > max_photos:
        rng = np.random.default_rng(seed)
        kept = [kept[i] for i in sorted(rng.choice(len(kept), max_photos, replace=False))]
    return kept


def download(images, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(images):
        path = out_dir / f"{im['id']}.jpg"
        if not path.exists():
            resp = requests.get(im["thumb_1024_url"], timeout=60)
            resp.raise_for_status()
            path.write_bytes(resp.content)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(images)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cities", nargs="+")
    parser.add_argument("--config", default="configs/cities.yaml")
    parser.add_argument("--out", default="data/cities")
    parser.add_argument("--max-photos", type=int, default=3000)
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("MAPILLARY_TOKEN")
    if not token:
        raise SystemExit("set MAPILLARY_TOKEN first")
    cfg = yaml.safe_load(open(args.config))

    for city in args.cities:
        c = cfg["cities"][city]
        images = list_images(token, *bbox(c["lat"], c["lon"], cfg["size_m"] / 2 - MARGIN_M))
        kept = thin(images, c["lat"], c["lon"], args.max_photos)
        panos = sum(bool(im.get("is_pano")) for im in kept)
        print(f"{city}: {len(images)} images, {len(kept)} after thinning "
              f"({panos} panoramas, {len(kept) - panos} perspective)")
        if args.count_only:
            continue

        city_dir = Path(args.out) / city
        download(kept, city_dir / "photos")
        with open(city_dir / "photos.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "lat", "lon", "compass", "is_pano", "captured_at", "sequence"])
            for im in kept:
                writer.writerow([im["id"], im["lat"], im["lon"], im.get("compass_angle"),
                                 bool(im.get("is_pano")), im.get("captured_at"), im.get("sequence")])


if __name__ == "__main__":
    main()
