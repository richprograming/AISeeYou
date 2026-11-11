"""Fine-tune a photo encoder and a map encoder with InfoNCE.

python -m scripts.train --config configs/train.yaml [--name run1] [--set freeze_blocks=8 epochs=6]

Writes outputs/<name>/{config.yaml, log.jsonl, best.pt}. Best = highest val R@10.

Each epoch is evaluated at three levels, to see where generalization breaks:
  seen      photos used in training (memorization)
  area      same training city, an eastern strip held out from training
  val       a city never used in training
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data import NearbyBatches, Pairs, load_city, select
from src.evaluate import embed_gallery, evaluate_city, evaluate_queries, pick_queries
from src.loss import info_nce
from src.model import Encoder


def photo_train_transform(model, size):
    return transforms.Compose([
        transforms.Resize(tuple(size)),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(model.mean, model.std),
    ])


def cosine_with_warmup(step, warmup, total):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def split_area(city, holdout_x_m, gap_m):
    """Photos west of holdout_x_m - gap_m train; photos east of holdout_x_m are held out."""
    x = city.xy[:, 0]
    return select(city, np.flatnonzero(x < holdout_x_m - gap_m)), select(city, np.flatnonzero(x > holdout_x_m))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--name", default="run")
    parser.add_argument("--root")
    parser.add_argument("--max-steps", type=int, help="stop early, for smoke tests")
    parser.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. freeze_blocks=8 epochs=6")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.root:
        cfg["root"] = args.root
    for item in args.set:
        key, value = item.split("=", 1)
        cfg[key] = yaml.safe_load(value)
    ev = cfg | cfg["eval"]
    splits = yaml.safe_load(open(cfg["cities_config"]))["splits"]
    train_names = [n for s in cfg["train_splits"] for n in splits[s]]

    out = Path("outputs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(out / "config.yaml", "w"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    photo_model = Encoder(cfg["backbone"]).to(device)
    map_model = Encoder(cfg["backbone"]).to(device)
    if cfg["freeze_blocks"]:
        photo_model.freeze(cfg["freeze_blocks"])
        map_model.freeze(cfg["freeze_blocks"])
    logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), device=device))

    cities, held_out = [], []
    for name in train_names:
        city = load_city(cfg["root"], name)
        if cfg["align"]:
            city = select(city, np.flatnonzero(~np.isnan(city.heading)))
        if not cfg["use_panos"]:
            city = pick_queries(city, "perspective", None)
        train, area = split_area(city, cfg["holdout_x_m"], cfg["radius_m"])
        cities.append(train)
        held_out.append(area)
    val_city = load_city(cfg["root"], splits["val"][0])
    print(f"train: {train_names} ({sum(len(c.photos) for c in cities)} photos), "
          f"held-out area: {sum(len(c.photos) for c in held_out)} photos, val: {val_city.name}")

    pairs = Pairs(cities, cfg["tile_m"], photo_train_transform(photo_model, cfg["photo_size"]),
                  map_model.transform(cfg["tile_size"]), cfg["jitter_m"], cfg["align"])
    sampler = NearbyBatches(cities, cfg["batch_size"], cfg["group"], cfg["near_m"], cfg["radius_m"])
    loader = DataLoader(pairs, batch_sampler=sampler, num_workers=cfg["num_workers"],
                        pin_memory=device == "cuda")

    params = [p for p in [*photo_model.parameters(), *map_model.parameters()] if p.requires_grad] + [logit_scale]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total = args.max_steps or cfg["epochs"] * len(sampler)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_with_warmup(s, cfg["warmup_steps"], total))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    step, best = 0, -1.0
    log = open(out / "log.jsonl", "a")
    for epoch in range(cfg["epochs"]):
        photo_model.train(), map_model.train()
        sampler.set_epoch(epoch)
        t0, running, count = time.time(), 0.0, 0
        for photo, tile, xy, city in loader:
            photo, tile, xy, city = (t.to(device, non_blocking=True) for t in (photo, tile, xy, city))
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                loss = info_nce(photo_model(photo), map_model(tile), logit_scale, xy, city, cfg["radius_m"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            with torch.no_grad():
                logit_scale.clamp_(0, math.log(100))

            step += 1
            running, count = running + loss.item(), count + 1
            if step % cfg["log_every"] == 0:
                print(f"epoch {epoch} step {step} loss {running / count:.3f} "
                      f"temp {1 / logit_scale.exp().item():.3f} {time.time() - t0:.0f}s", flush=True)
                running, count = 0.0, 0
            if args.max_steps and step >= args.max_steps:
                break

        photo_model.eval(), map_model.eval()
        train_loss_time = time.time() - t0
        # seen photos and held-out area share the first training city's gallery
        first = cities[0]
        centers, gallery = embed_gallery(map_model, first, ev, device)
        seen = pick_queries(first, ev["photo_type"], ev["max_queries"], ev["seed"])
        area = pick_queries(held_out[0], ev["photo_type"], ev["max_queries"], ev["seed"])
        record = {
            "epoch": epoch, "step": step,
            "seen": evaluate_queries(photo_model, seen, centers, gallery, ev, device),
            "area": evaluate_queries(photo_model, area, centers, gallery, ev, device),
            "val": evaluate_city(photo_model, map_model, val_city, ev, device),
            "train_seconds": round(train_loss_time, 1), "seconds": round(time.time() - t0, 1),
        }
        print(json.dumps(record), flush=True)
        log.write(json.dumps(record) + "\n")
        log.flush()
        if record["val"]["R@10"] > best:
            best = record["val"]["R@10"]
            torch.save({"photo": photo_model.state_dict(), "map": map_model.state_dict(),
                        "logit_scale": logit_scale.detach().cpu(), "config": cfg, "epoch": epoch},
                       out / "best.pt")
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
