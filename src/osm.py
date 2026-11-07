import numpy as np
from PIL import Image, ImageDraw

from src.geo import to_xy

# one flat color per class; the model only needs them to be distinct
COLORS = {
    "background": (242, 239, 233),
    "sand": (240, 220, 160),
    "green": (172, 210, 156),
    "water": (120, 170, 220),
    "parking": (205, 205, 205),
    "rail": (120, 80, 140),
    "road": (80, 80, 80),
    "path": (200, 130, 90),
    "building": (215, 100, 80),
    "tree": (60, 140, 60),
}
DRAW_ORDER = ["sand", "green", "water", "parking", "rail", "road", "path", "building", "tree"]

ROAD_WIDTH_M = {
    "motorway": 14, "trunk": 14, "primary": 12, "secondary": 10, "tertiary": 9,
    "residential": 7, "unclassified": 7, "living_street": 6, "service": 4,
}
PATH_TYPES = {"footway", "path", "pedestrian", "cycleway", "steps", "track"}
GREEN = {"park", "garden", "pitch", "playground", "grass", "forest", "meadow",
         "recreation_ground", "village_green", "wood", "scrub", "grassland"}


def classify(tags):
    if "building" in tags:
        return "building"
    if tags.get("highway") in ROAD_WIDTH_M:
        return "road"
    if tags.get("highway") in PATH_TYPES:
        return "path"
    if tags.get("railway") in {"rail", "tram", "light_rail", "subway"}:
        return "rail"
    if tags.get("natural") == "water" or "waterway" in tags:
        return "water"
    if tags.get("natural") == "tree":
        return "tree"
    if tags.get("amenity") == "parking":
        return "parking"
    if tags.get("natural") in {"beach", "sand"}:
        return "sand"
    if GREEN & {tags.get("leisure"), tags.get("landuse"), tags.get("natural")}:
        return "green"
    return None


def line_width_m(kind, tags):
    if kind == "road":
        return ROAD_WIDTH_M[tags["highway"]]
    return {"path": 2, "rail": 3, "water": 4}.get(kind, 0)


def shapes(elements):
    """Yield (kind, tags, rings) where rings are lists of (lat, lon) and nodes are 1-point rings.
    Relations keep only their closed member ways; holes are ignored."""
    for el in elements:
        tags = el.get("tags", {})
        kind = classify(tags)
        if kind is None:
            continue
        if el["type"] == "node":
            yield kind, tags, [[(el["lat"], el["lon"])]]
        elif el["type"] == "way":
            yield kind, tags, [[(p["lat"], p["lon"]) for p in el["geometry"]]]
        elif el["type"] == "relation":
            ways = [[(p["lat"], p["lon"]) for p in m["geometry"]]
                    for m in el.get("members", []) if m.get("role") == "outer" and "geometry" in m]
            yield kind, tags, join_rings(ways)


def join_rings(ways):
    """Chain open ways that share endpoints into closed rings (large lakes and rivers come split)."""
    rings, todo = [], [list(w) for w in ways if len(w) > 1]
    while todo:
        ring = todo.pop()
        grew = True
        while ring[0] != ring[-1] and grew:
            grew = False
            for i, w in enumerate(todo):
                if w[0] == ring[-1]:
                    ring = ring + w[1:]
                elif w[-1] == ring[-1]:
                    ring = ring + w[::-1][1:]
                elif w[-1] == ring[0]:
                    ring = w + ring[1:]
                elif w[0] == ring[0]:
                    ring = w[::-1] + ring[1:]
                else:
                    continue
                todo.pop(i)
                grew = True
                break
        if ring[0] == ring[-1]:
            rings.append(ring)
    return rings


def render(elements, lat0, lon0, size_m, m_per_px=0.5):
    size_px = round(size_m / m_per_px)
    img = Image.new("RGB", (size_px, size_px), COLORS["background"])
    draw = ImageDraw.Draw(img)

    def to_px(ring):
        xy = to_xy([p[0] for p in ring], [p[1] for p in ring], lat0, lon0)
        cols = (xy[:, 0] + size_m / 2) / m_per_px
        rows = (size_m / 2 - xy[:, 1]) / m_per_px
        return list(zip(cols.tolist(), rows.tolist()))

    items = sorted(shapes(elements), key=lambda s: DRAW_ORDER.index(s[0]))
    for kind, tags, rings in items:
        color = COLORS[kind]
        for ring in rings:
            pts = to_px(ring)
            if kind == "tree":
                c, r = pts[0], 3 / m_per_px
                draw.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=color)
            elif len(pts) > 2 and ring[0] == ring[-1] and kind not in ("road", "path", "rail"):
                draw.polygon(pts, fill=color)
            elif len(pts) > 1:
                width = max(1, round(line_width_m(kind, tags) / m_per_px))
                draw.line(pts, fill=color, width=width, joint="curve")
    return np.asarray(img)
