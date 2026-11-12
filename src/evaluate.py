import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import Photos, Tiles, grid, select
from src.metrics import retrieval_metrics, search_mask, topk
from src.model import embed


def pick_queries(city, photo_type, max_queries, seed=0):
    keep = {"pano": city.is_pano, "perspective": ~city.is_pano}.get(photo_type, np.ones_like(city.is_pano))
    idx = np.flatnonzero(keep)
    if max_queries and len(idx) > max_queries:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_queries, replace=False))
    return select(city, idx)


def loader(ds, cfg):
    return DataLoader(ds, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])


def embed_gallery(map_model, city, cfg, device):
    """Grid of tiles over the whole city area, each at cfg['rotations'] headings."""
    centers = grid(city.size_m, cfg["tile_m"], cfg["stride_m"])
    rotations = cfg.get("rotations", 1)
    tiles = Tiles(city, centers, cfg["tile_m"], map_model.transform(cfg["tile_size"]), rotations)
    return centers, embed(map_model, loader(tiles, cfg), device, log_every=0)


def evaluate_queries(photo_model, queries, centers, gallery, cfg, device):
    """Heading of the query photos is never used: a location scores as its best rotation."""
    photos = Photos(queries.photos, photo_model.transform(cfg["photo_size"]))
    query = embed(photo_model, loader(photos, cfg), device, log_every=0)
    top = topk(query, gallery, rotations=cfg.get("rotations", 1))
    metrics = retrieval_metrics(top, queries.xy, centers, cfg["radius_m"])
    metrics.update(queries=len(queries.photos), gallery=len(centers))
    if cfg.get("search_m"):
        mask = search_mask(queries.xy, centers, cfg["search_m"], cfg["seed"])
        top = topk(query, gallery, rotations=cfg.get("rotations", 1), mask=torch.from_numpy(mask))
        metrics["local"] = retrieval_metrics(top, queries.xy, centers, cfg["radius_m"], mask=mask)
    return metrics


def evaluate_city(photo_model, map_model, city, cfg, device):
    queries = pick_queries(city, cfg["photo_type"], cfg["max_queries"], cfg["seed"])
    centers, gallery = embed_gallery(map_model, city, cfg, device)
    return evaluate_queries(photo_model, queries, centers, gallery, cfg, device)
