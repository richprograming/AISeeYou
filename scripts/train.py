"""Fine-tune a photo encoder and a map encoder with InfoNCE.

python -m scripts.train --config configs/train.yaml [--name run1] [--max-steps 5]

Writes outputs/<name>/{config.yaml, log.jsonl, best.pt}. Best = highest val R@10.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data import NearbyBatches, Pairs, load_city
from src.evaluate import evaluate_city, pick_queries
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--name", default="run")
    parser.add_argument("--root")
    parser.add_argument("--max-steps", type=int, help="stop early, for smoke tests")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.root:
        cfg["root"] = args.root
    splits = yaml.safe_load(open(cfg["cities_config"]))["splits"]
    train_names = [n for s in cfg["train_splits"] for n in splits[s]]
    val_name = splits["val"][0]

    out = Path("outputs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(out / "config.yaml", "w"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    photo_model = Encoder(cfg["backbone"]).to(device)
    map_model = Encoder(cfg["backbone"]).to(device)
    logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), device=device))

    cities = [load_city(cfg["root"], n) for n in train_names]
    if not cfg["use_panos"]:
        cities = [pick_queries(c, "perspective", None) for c in cities]
    val_city = load_city(cfg["root"], val_name)
    print(f"train: {train_names} ({sum(len(c.photos) for c in cities)} photos), val: {val_name}")

    pairs = Pairs(cities, cfg["tile_m"], photo_train_transform(photo_model, cfg["photo_size"]),
                  map_model.transform(cfg["tile_size"]), cfg["jitter_m"])
    sampler = NearbyBatches(cities, cfg["batch_size"], cfg["group"], cfg["near_m"], cfg["radius_m"])
    loader = DataLoader(pairs, batch_sampler=sampler, num_workers=cfg["num_workers"], pin_memory=device == "cuda")

    params = [*photo_model.parameters(), *map_model.parameters(), logit_scale]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total = args.max_steps or cfg["epochs"] * len(sampler)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_with_warmup(s, cfg["warmup_steps"], total))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    step, best = 0, -1.0
    log = open(out / "log.jsonl", "a")
    for epoch in range(cfg["epochs"]):
        photo_model.train(), map_model.train()
        sampler.set_epoch(epoch)
        t0, running = time.time(), 0.0
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
            running += loss.item()
            if step % cfg["log_every"] == 0:
                print(f"epoch {epoch} step {step} loss {running / cfg['log_every']:.3f} "
                      f"temp {1 / logit_scale.exp().item():.3f} {time.time() - t0:.0f}s", flush=True)
                running = 0.0
            if args.max_steps and step >= args.max_steps:
                break

        photo_model.eval(), map_model.eval()
        val = evaluate_city(photo_model, map_model, val_city, cfg | cfg["eval"], device)
        record = {"epoch": epoch, "step": step, "val": val, "seconds": round(time.time() - t0, 1)}
        print(json.dumps(record), flush=True)
        log.write(json.dumps(record) + "\n")
        log.flush()
        if val["R@10"] > best:
            best = val["R@10"]
            torch.save({"photo": photo_model.state_dict(), "map": map_model.state_dict(),
                        "logit_scale": logit_scale.detach().cpu(), "config": cfg, "epoch": epoch},
                       out / "best.pt")
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
