import torch
import torch.nn.functional as F


def info_nce(photo, tile, logit_scale, xy=None, city=None, radius_m=50):
    """Symmetric InfoNCE: photo i should match tile i and no other tile in the batch.

    photo, tile: (B, D) L2-normalized embeddings.
    If xy (B, 2) and city (B,) are given, off-diagonal pairs closer than radius_m in the same
    city are masked out: they show the same place, so they are not real negatives.
    """
    logits = logit_scale.exp() * photo @ tile.T
    if xy is not None:
        same_place = (torch.cdist(xy, xy) < radius_m) & (city[:, None] == city[None, :])
        same_place.fill_diagonal_(False)
        logits = logits.masked_fill(same_place, float("-inf"))
    target = torch.arange(len(photo), device=photo.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
