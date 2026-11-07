"""Zero-shot retrieval baseline: pretrained backbone, no training.

python -m scripts.baseline --config configs/baseline.yaml [--root ...] [--max-queries N]
"""
import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data import ImageList, load_vigor, subset
from src.metrics import rank_queries, retrieval_metrics
from src.model import Encoder, embed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--root")
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--full-gallery", action="store_true")
    parser.add_argument("--out", default="outputs/baseline.json")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.root:
        cfg["root"] = args.root
    if args.max_queries:
        cfg["max_queries"] = args.max_queries
    if args.full_gallery:
        cfg["full_gallery"] = True

    data = load_vigor(cfg["root"], cfg["cities"], cfg["split_file"], cfg["label_dir"])
    print(f"loaded {len(data.panos)} panoramas, {len(data.sats)} satellites")
    if cfg["max_queries"]:
        data = subset(data, cfg["max_queries"], cfg["seed"], cfg["full_gallery"])
    print(f"evaluating {len(data.panos)} queries against {len(data.sats)} gallery images")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Encoder(cfg["backbone"]).eval().to(device)

    def loader(paths, size):
        ds = ImageList(paths, model.transform(size))
        return DataLoader(ds, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])

    t0 = time.time()
    query = embed(model, loader(data.panos, cfg["pano_size"]), device)
    gallery = embed(model, loader(data.sats, cfg["sat_size"]), device)
    ranks, top1 = rank_queries(query, gallery, data.labels[:, 0])
    metrics = retrieval_metrics(ranks, top1, data.labels, data.pano_coords(), data.sat_coords())
    metrics.update(queries=len(data.panos), gallery=len(data.sats), seconds=round(time.time() - t0, 1))

    print(json.dumps(metrics, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": cfg, "metrics": metrics}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
