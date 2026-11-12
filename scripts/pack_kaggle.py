"""Zip data/cities for upload as a Kaggle dataset (photos, csv and rendered maps only).

python -m scripts.pack_kaggle --user <kaggle-username>
cd data/kaggle
kaggle datasets create -p .                       # first time
kaggle datasets version -p . -m "more cities"     # later

Run kaggle from inside data/kaggle: on Windows the CLI breaks on paths like "data/kaggle".
"""
import argparse
import csv
import json
import zipfile
from pathlib import Path

KEEP = {"map.png", "map.json", "photos.csv"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    parser.add_argument("--src", default="data/cities")
    parser.add_argument("--out", default="data/kaggle")
    args = parser.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out / "cities.zip", "w", zipfile.ZIP_STORED) as zf:  # jpg/png are already compressed
        for city in sorted(p for p in src.iterdir() if (p / "photos.csv").exists()):
            # only photos listed in the csv: a re-fetch can leave older downloads in the folder
            with open(city / "photos.csv") as f:
                ids = [row["id"] for row in csv.DictReader(f)]
            files = [city / n for n in KEEP] + [city / "photos" / f"{i}.jpg" for i in ids]
            for f in files:
                zf.write(f, f.relative_to(src))
            print(f"{city.name}: {len(files) - len(KEEP)} photos")

    meta = {
        "title": "AISeeYou cities",
        "id": f"{args.user}/aiseeyou-cities",
        "licenses": [{"name": "CC-BY-SA-4.0"}],
        "subtitle": "Mapillary street photos (CC BY-SA 4.0) with OpenStreetMap rasters (ODbL)",
    }
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out / 'cities.zip'} ({(out / 'cities.zip').stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
