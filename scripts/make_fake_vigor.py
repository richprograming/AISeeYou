"""Tiny fake dataset with the VIGOR folder layout, to test the pipeline without the real data.

Each panorama is a noisy, stretched copy of its positive satellite tile, so pixel features
retrieve it perfectly if the labels are wired correctly (see tests/test_pipeline.py).
Pretrained backbones score near chance here: random textures all embed alike.

python -m scripts.make_fake_vigor data/fake_vigor --n 64
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

CITY = "Chicago"
ORIGIN = (41.8781, -87.6298)
STEP = 0.0007  # ~70 m between tile centers


def make_fake_vigor(root, n=64, seed=0):
    rng = np.random.default_rng(seed)
    root = Path(root)
    sat_dir = root / CITY / "satellite"
    pano_dir = root / CITY / "panorama"
    split_dir = root / "splits" / CITY
    for d in (sat_dir, pano_dir, split_dir):
        d.mkdir(parents=True, exist_ok=True)

    side = int(np.ceil(np.sqrt(n)))
    sats, coords = [], []
    for i in range(n):
        lat = ORIGIN[0] + (i // side) * STEP
        lon = ORIGIN[1] + (i % side) * STEP
        name = f"satellite_{lat:.7f}_{lon:.7f}.png"
        # random blocky texture: upsampled low-res color noise
        small = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
        Image.fromarray(small).resize((256, 256), Image.NEAREST).save(sat_dir / name)
        sats.append(name)
        coords.append((lat, lon))

    lines = []
    for i, (lat, lon) in enumerate(coords):
        plat = lat + rng.uniform(-0.2, 0.2) * STEP
        plon = lon + rng.uniform(-0.2, 0.2) * STEP
        pano_name = f"fake{i:05d},{plat:.6f},{plon:.6f},.jpg"
        sat = np.asarray(Image.open(sat_dir / sats[i]).resize((512, 256)), dtype=np.float32)
        noisy = np.clip(sat + rng.normal(0, 25, sat.shape), 0, 255).astype(np.uint8)
        Image.fromarray(noisy).save(pano_dir / pano_name)

        semis = rng.choice([j for j in range(n) if j != i], size=3, replace=False)
        refs = [i, *semis]
        tokens = [pano_name]
        for j in refs:
            tokens += [sats[j], f"{rng.normal(0, 50):.4f}", f"{rng.normal(0, 50):.4f}"]
        lines.append(" ".join(tokens))

    (split_dir / "satellite_list.txt").write_text("\n".join(sats) + "\n")
    for f in ("same_area_balanced_train.txt", "same_area_balanced_test.txt", "pano_label_balanced.txt"):
        (split_dir / f).write_text("\n".join(lines) + "\n")
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--n", type=int, default=64)
    args = parser.parse_args()
    make_fake_vigor(args.root, args.n)
    print(f"wrote fake VIGOR to {args.root}")
