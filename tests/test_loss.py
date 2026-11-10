import math

import torch
import torch.nn.functional as F

from src.loss import info_nce

SCALE = torch.tensor(math.log(10.0))


def test_aligned_pairs_have_lower_loss():
    a = F.normalize(torch.randn(8, 16), dim=-1)
    shuffled = a[torch.randperm(8)]
    assert info_nce(a, a, SCALE) < info_nce(a, shuffled, SCALE)


def test_same_place_pairs_are_masked():
    # photo 0 and 1 are 10 m apart: tile 1 must not count as a negative for photo 0
    photo = F.normalize(torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), dim=-1)
    tile = photo.clone()
    xy = torch.tensor([[0.0, 0.0], [10.0, 0.0], [500.0, 0.0]])
    city = torch.tensor([0, 0, 0])
    masked = info_nce(photo, tile, SCALE, xy, city)
    unmasked = info_nce(photo, tile, SCALE)
    assert masked < unmasked


def test_other_city_is_not_masked():
    photo = F.normalize(torch.tensor([[1.0, 0.0], [1.0, 0.0]]), dim=-1)
    xy = torch.zeros(2, 2)
    assert info_nce(photo, photo, SCALE, xy, torch.tensor([0, 1])) == info_nce(photo, photo, SCALE)
