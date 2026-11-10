# AISeeYou

Cross-view geolocalization: given a street-level photo, find where it was taken by matching it
against map tiles rendered from OpenStreetMap. Work in progress.

Data is fully open: photos from Mapillary (CC BY-SA 4.0), maps from OpenStreetMap (ODbL).

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Data

Cities are defined in `configs/cities.yaml` (a 2 x 2 km square each). For each city:

```
python -m scripts.fetch_osm porto_alegre                       # map raster, no account needed
set MAPILLARY_TOKEN=...                                        # free client token
python -m scripts.fetch_mapillary porto_alegre --count-only    # coverage check
python -m scripts.fetch_mapillary porto_alegre --max-photos 3000
```

Result: `data/cities/<city>/{map.png, map.json, photos.csv, photos/}`.

## Baseline

```
python -m scripts.baseline --cities porto_alegre
```

Each photo is ranked against a grid of 100 m map tiles (25 m stride). A retrieved tile counts as
correct if its center is within 50 m of the photo. Chance level is reported next to each result.

Map data (c) OpenStreetMap contributors. Imagery from Mapillary contributors.
