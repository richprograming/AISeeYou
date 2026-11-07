"""Download OSM features for each city area from the Overpass API and render a map raster.

python -m scripts.fetch_osm sao_paulo [berlin ...] [--out data/cities]

Writes data/cities/<city>/osm.json (raw), map.png and map.json (georeference).
"""
import argparse
import json
from pathlib import Path

import requests
import yaml
from PIL import Image

from src.geo import bbox
from src.osm import render

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
QUERY = """
[out:json][timeout:300];
(
  way["building"]({b}); relation["building"]({b});
  way["highway"]({b});
  way["railway"]({b});
  way["natural"~"water|wood|scrub|grassland|beach|sand"]({b});
  relation["natural"~"water|wood|beach"]({b});
  way["waterway"]({b});
  way["landuse"~"grass|forest|meadow|recreation_ground|village_green"]({b});
  way["leisure"~"park|garden|pitch|playground"]({b}); relation["leisure"="park"]({b});
  way["amenity"="parking"]({b});
  node["natural"="tree"]({b});
);
out geom;
"""


def fetch(city, lat, lon, size_m, out):
    path = out / city / "osm.json"
    if path.exists():
        print(f"{city}: using cached {path}")
        return json.loads(path.read_text())["elements"]
    s, w, n, e = bbox(lat, lon, size_m / 2)
    query = QUERY.replace("{b}", f"{s},{w},{n},{e}")
    resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=360,
                         headers={"User-Agent": "aiseeyou-research/0.1"})
    resp.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    print(f"{city}: {len(resp.content) / 1e6:.1f} MB from Overpass")
    return resp.json()["elements"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cities", nargs="+")
    parser.add_argument("--config", default="configs/cities.yaml")
    parser.add_argument("--out", default="data/cities")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out = Path(args.out)
    for city in args.cities:
        c = cfg["cities"][city]
        elements = fetch(city, c["lat"], c["lon"], cfg["size_m"], out)
        raster = render(elements, c["lat"], c["lon"], cfg["size_m"], cfg["m_per_px"])
        Image.fromarray(raster).save(out / city / "map.png")
        meta = {"lat": c["lat"], "lon": c["lon"], "size_m": cfg["size_m"], "m_per_px": cfg["m_per_px"]}
        (out / city / "map.json").write_text(json.dumps(meta, indent=2))
        print(f"{city}: {len(elements)} elements, map {raster.shape[1]}x{raster.shape[0]} px")


if __name__ == "__main__":
    main()
