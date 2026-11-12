"""Evaluate a trained checkpoint on cities never used in training.

python -m scripts.test outputs/latam/best.pt [--cities porto_alegre ...] [--out outputs/test]

Writes <out>/metrics.json and, per city, <out>/<city>.npz with the query and gallery
embeddings (used by the error analysis and the demo).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data import load_city
from src.evaluate import embed_gallery, embed_queries, pick_queries, score
from src.model import Encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--root", default="data/cities")
    parser.add_argument("--cities", nargs="+", help="default: the test split")
    parser.add_argument("--out", default="outputs/test")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    ev = cfg | cfg["eval"] | {"num_workers": 2, "batch_size": 32}
    cities = args.cities or yaml.safe_load(open(cfg["cities_config"]))["splits"]["test"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    photo_model = Encoder(cfg["backbone"], pretrained=False)
    map_model = Encoder(cfg["backbone"], pretrained=False)
    photo_model.load_state_dict(ckpt["photo"])
    map_model.load_state_dict(ckpt["map"])
    photo_model.eval().to(device), map_model.eval().to(device)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in cities:
        t0 = time.time()
        city = load_city(args.root, name)
        queries = pick_queries(city, ev["photo_type"], ev["max_queries"], ev["seed"])
        centers, gallery = embed_gallery(map_model, city, ev, device)
        query = embed_queries(photo_model, queries, ev, device)
        results[name] = score(query, queries.xy, centers, gallery, ev)
        results[name]["seconds"] = round(time.time() - t0, 1)
        print(name, json.dumps(results[name]), flush=True)
        np.savez(out / f"{name}.npz", query=query.numpy().astype(np.float16),
                 gallery=gallery.numpy().astype(np.float16), centers=centers, query_xy=queries.xy,
                 photos=[p.name for p in queries.photos], heading=queries.heading)
        json.dump({"checkpoint": args.checkpoint, "epoch": ckpt["epoch"], "eval": cfg["eval"],
                   "results": results}, open(out / "metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
