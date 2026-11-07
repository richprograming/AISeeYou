import numpy as np

from src.geo import bbox, to_latlon, to_xy


def test_one_degree_latitude():
    x, y = to_xy(1.0, 0.0, 0.0, 0.0)
    assert abs(x) < 1e-6 and abs(y - 111_195) < 10


def test_roundtrip():
    lat, lon = to_latlon(300.0, -450.0, -23.56, -46.65)
    assert np.allclose(to_xy(lat, lon, -23.56, -46.65), [300, -450], atol=1e-6)


def test_bbox_is_square_in_meters():
    s, w, n, e = bbox(52.5, 13.4, 1000)
    assert np.allclose(to_xy(s, w, 52.5, 13.4), [-1000, -1000], atol=1e-6)
    assert np.allclose(to_xy(n, e, 52.5, 13.4), [1000, 1000], atol=1e-6)
