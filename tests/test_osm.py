from src.geo import to_latlon
from src.osm import COLORS, join_rings, render


def test_join_rings_reverses_and_chains():
    a, b, c = (0, 0), (0, 1), (1, 1)
    rings = join_rings([[a, b], [c, b], [c, a]])
    assert len(rings) == 1 and rings[0][0] == rings[0][-1] and len(rings[0]) == 4


def test_open_ring_is_dropped():
    assert join_rings([[(0, 0), (0, 1)]]) == []


def test_building_drawn_at_its_position():
    corners = [to_latlon(x, y, 10.0, 20.0) for x, y in [(20, 20), (40, 20), (40, 40), (20, 40), (20, 20)]]
    way = {"type": "way", "tags": {"building": "yes"},
           "geometry": [{"lat": float(lat), "lon": float(lon)} for lat, lon in corners]}
    img = render([way], 10.0, 20.0, size_m=100, m_per_px=1.0)
    assert tuple(img[100 // 2 - 30, 100 // 2 + 30]) == COLORS["building"]  # x=30, y=30
    assert tuple(img[50, 50]) == COLORS["background"]
