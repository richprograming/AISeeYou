"""Zero-shot retrieval baseline: pretrained backbone, no training.

For each city, every photo is matched against a grid of map tiles covering the city area.

python -m scripts.baseline --config configs/baseline.yaml [--root ...] [--max-queries N]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data import Photos, Tiles, grid, load_city, select
from src.metrics import retrieval_metrics, topk
from src.model import Encoder, embed


def pick_queries(city, photo_type, max_queries, seed):
    keep = {"pano": city.is_pano, "perspective": ~city.is_pano}.get(photo_type, np.ones_like(city.is_pano))
    idx = np.flatnonzero(keep)
    if max_queries and len(idx) > max_queries:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_queries, replace=False))
    return select(city, idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--root")
    parser.add_argument("--cities", nargs="+")
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--stride-m", type=float)
    parser.add_argument("--out", default="outputs/baseline.json")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    for key in ("root", "cities", "max_queries", "stride_m"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Encoder(cfg["backbone"]).eval().to(device)

    def loader(ds):
        return DataLoader(ds, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])

    results = {}
    for name in cfg["cities"]:
        t0 = time.time()
        city = pick_queries(load_city(cfg["root"], name), cfg["photo_type"], cfg["max_queries"], cfg["seed"])
        centers = grid(city.size_m, cfg["tile_m"], cfg["stride_m"])
        print(f"{name}: {len(city.photos)} queries, {len(centers)} gallery tiles")

        query = embed(model, loader(Photos(city.photos, model.transform(cfg["photo_size"]))), device)
        gallery = embed(model, loader(Tiles(city, centers, cfg["tile_m"], model.transform(cfg["tile_size"]))), device)
        metrics = retrieval_metrics(topk(query, gallery), city.xy, centers, cfg["radius_m"])
        metrics.update(queries=len(city.photos), gallery=len(centers), seconds=round(time.time() - t0, 1))
        print(json.dumps(metrics, indent=2))
        results[name] = metrics

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": cfg, "results": results}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
