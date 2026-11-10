import numpy as np
from torch.utils.data import DataLoader

from src.data import Photos, Tiles, grid, select
from src.metrics import retrieval_metrics, topk
from src.model import embed


def pick_queries(city, photo_type, max_queries, seed=0):
    keep = {"pano": city.is_pano, "perspective": ~city.is_pano}.get(photo_type, np.ones_like(city.is_pano))
    idx = np.flatnonzero(keep)
    if max_queries and len(idx) > max_queries:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_queries, replace=False))
    return select(city, idx)


def evaluate_city(photo_model, map_model, city, cfg, device):
    """Rank every query photo against a grid of map tiles covering the whole city area."""
    queries = pick_queries(city, cfg["photo_type"], cfg["max_queries"], cfg["seed"])
    centers = grid(city.size_m, cfg["tile_m"], cfg["stride_m"])

    def loader(ds):
        return DataLoader(ds, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])

    photos = Photos(queries.photos, photo_model.transform(cfg["photo_size"]))
    tiles = Tiles(city, centers, cfg["tile_m"], map_model.transform(cfg["tile_size"]))
    query = embed(photo_model, loader(photos), device, log_every=0)
    gallery = embed(map_model, loader(tiles), device, log_every=0)
    metrics = retrieval_metrics(topk(query, gallery), queries.xy, centers, cfg["radius_m"])
    metrics.update(queries=len(queries.photos), gallery=len(centers))
    return metrics
