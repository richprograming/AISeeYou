import numpy as np

EARTH_RADIUS_M = 6_371_000


def to_xy(lat, lon, lat0, lon0):
    """Local metric coordinates (x east, y north) around (lat0, lon0). Fine for a few km."""
    lat, lon = np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)
    x = np.radians(lon - lon0) * np.cos(np.radians(lat0)) * EARTH_RADIUS_M
    y = np.radians(lat - lat0) * EARTH_RADIUS_M
    return np.stack([x, y], axis=-1)


def to_latlon(x, y, lat0, lon0):
    lat = lat0 + np.degrees(np.asarray(y) / EARTH_RADIUS_M)
    lon = lon0 + np.degrees(np.asarray(x) / (EARTH_RADIUS_M * np.cos(np.radians(lat0))))
    return lat, lon


def bbox(lat0, lon0, half_m):
    """(south, west, north, east) of a square of side 2 * half_m."""
    s, w = to_latlon(-half_m, -half_m, lat0, lon0)
    n, e = to_latlon(half_m, half_m, lat0, lon0)
    return float(s), float(w), float(n), float(e)
