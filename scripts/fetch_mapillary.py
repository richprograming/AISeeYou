"""Download street-level photos from Mapillary inside each city area.

Needs a free client token: MAPILLARY_TOKEN in the environment or in .env.

python -m scripts.fetch_mapillary sao_paulo --count-only     # coverage check, no images
python -m scripts.fetch_mapillary sao_paulo --max-photos 3000

Coverage comes from Mapillary's vector tiles (the bbox search endpoint times out in dense
areas). Only the photos kept after thinning get a metadata lookup and a download.
Writes data/cities/<city>/photos.csv and photos/<id>.jpg (1024 px wide thumbnails).
"""
import argparse
import csv
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mapbox_vector_tile
import numpy as np
import requests
import yaml

from src.geo import bbox, to_xy

TILE_URL = "https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}"
GRAPH_URL = "https://graph.mapillary.com/"
ZOOM = 14           # the zoom level where the image layer is available
MARGIN_M = 50       # keep photos away from the map border so tiles around them exist
THIN_M = 10         # at most one photo per 10 m cell, to avoid near duplicates


def read_token(env_file=".env"):
    if os.environ.get("MAPILLARY_TOKEN"):
        return os.environ["MAPILLARY_TOKEN"]
    if Path(env_file).exists():
        for line in Path(env_file).read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "MAPILLARY_TOKEN":
                return value.strip()
    raise SystemExit("set MAPILLARY_TOKEN in the environment or in .env")


def get(url, **kwargs):
    # never let an exception print the url: the tile endpoint needs the token in it
    resp = requests.get(url, timeout=60, **kwargs)
    if not resp.ok:
        raise SystemExit(f"HTTP {resp.status_code} from {url.split('?')[0]}: {resp.text[:200]}")
    return resp


def tile_xy(lat, lon, z=ZOOM):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def tile_lonlat(x, y, z=ZOOM):
    n = 2 ** z
    lon = x / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def list_images(token, s, w, n, e):
    x0, y0 = tile_xy(n, w)
    x1, y1 = tile_xy(s, e)
    images = []
    for tx in range(int(x0), int(x1) + 1):
        for ty in range(int(y0), int(y1) + 1):
            resp = get(TILE_URL.format(z=ZOOM, x=tx, y=ty), params={"access_token": token})
            layer = mapbox_vector_tile.decode(resp.content, default_options={"y_coord_down": True})
            layer = layer.get("image", {"features": [], "extent": 4096})
            for f in layer["features"]:
                px, py = f["geometry"]["coordinates"]
                lat, lon = tile_lonlat(tx + px / layer["extent"], ty + py / layer["extent"])
                if s <= lat <= n and w <= lon <= e:
                    images.append({**f["properties"], "lat": lat, "lon": lon})
    return images


def thin(images, lat0, lon0, max_photos, seed=0):
    # newest first, so photos match the current OSM map as closely as possible
    images = sorted(images, key=lambda im: im.get("captured_at", 0), reverse=True)
    kept, seen = [], set()
    for im in images:
        x, y = to_xy(im["lat"], im["lon"], lat0, lon0)
        cell = (int(x // THIN_M), int(y // THIN_M))
        if cell not in seen:
            seen.add(cell)
            kept.append(im)
    if max_photos and len(kept) > max_photos:
        rng = np.random.default_rng(seed)
        kept = [kept[i] for i in sorted(rng.choice(len(kept), max_photos, replace=False))]
    return kept


def add_details(token, images, batch=50):
    """Thumbnail url and the refined (structure-from-motion) position, when Mapillary has one."""
    session_headers = {"Authorization": f"OAuth {token}"}
    out = []
    for i in range(0, len(images), batch):
        chunk = images[i:i + batch]
        params = {"ids": ",".join(str(im["id"]) for im in chunk), "fields": "id,computed_geometry,thumb_1024_url"}
        details = get(GRAPH_URL, params=params, headers=session_headers).json()
        for im in chunk:
            d = details.get(str(im["id"]), {})
            if "thumb_1024_url" not in d:
                continue
            if "computed_geometry" in d:
                im["lon"], im["lat"] = d["computed_geometry"]["coordinates"]
            out.append({**im, "url": d["thumb_1024_url"]})
    return out


def download_one(im, out_dir):
    path = out_dir / f"{im['id']}.jpg"
    if path.exists():
        return True
    try:
        resp = requests.get(im["url"], timeout=60)
    except requests.RequestException:
        return False
    if resp.ok:
        path.write_bytes(resp.content)
    return resp.ok


def download(images, out_dir, workers=4):
    """Returns the images that were saved; failed downloads are dropped."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (im, ok) in enumerate(zip(images, pool.map(lambda im: download_one(im, out_dir), images))):
            if ok:
                saved.append(im)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(images)}", flush=True)
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cities", nargs="+")
    parser.add_argument("--config", default="configs/cities.yaml")
    parser.add_argument("--out", default="data/cities")
    parser.add_argument("--max-photos", type=int, default=3000)
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    token = read_token()
    cfg = yaml.safe_load(open(args.config))

    for city in args.cities:
        c = cfg["cities"][city]
        images = list_images(token, *bbox(c["lat"], c["lon"], cfg["size_m"] / 2 - MARGIN_M))
        kept = thin(images, c["lat"], c["lon"], None if args.count_only else args.max_photos)
        panos = sum(bool(im.get("is_pano")) for im in kept)
        print(f"{city}: {len(images)} images, {len(kept)} after thinning "
              f"({panos} panoramas, {len(kept) - panos} perspective)", flush=True)
        if args.count_only:
            continue

        kept = add_details(token, kept)
        city_dir = Path(args.out) / city
        kept = download(kept, city_dir / "photos", args.workers)
        with open(city_dir / "photos.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "lat", "lon", "compass", "is_pano", "captured_at", "sequence", "quality"])
            for im in kept:
                writer.writerow([im["id"], im["lat"], im["lon"], im.get("compass_angle"), bool(im.get("is_pano")),
                                 im.get("captured_at"), im.get("sequence_id"), im.get("quality_score")])
        print(f"{city}: saved {len(kept)} photos", flush=True)


if __name__ == "__main__":
    main()
