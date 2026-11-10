"""Zero-shot retrieval baseline: pretrained backbone, no training.

For each city, every photo is matched against a grid of map tiles covering the city area.

python -m scripts.baseline --config configs/baseline.yaml [--cities ...] [--stride-m 50]
"""
import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from src.data import load_city
from src.evaluate import evaluate_city
from src.model import Encoder


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

    results = {}
    for name in cfg["cities"]:
        t0 = time.time()
        metrics = evaluate_city(model, model, load_city(cfg["root"], name), cfg, device)
        metrics["seconds"] = round(time.time() - t0, 1)
        print(name, json.dumps(metrics, indent=2))
        results[name] = metrics

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": cfg, "results": results}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
