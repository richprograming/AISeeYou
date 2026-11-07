from src.data import load_vigor, pano_latlon, sat_latlon, subset
from scripts.make_fake_vigor import make_fake_vigor


def test_filename_coords():
    assert pano_latlon("-Bm25YXcr-aJ-Nbdz0oIw,47.593046,-122.300124,.jpg") == (47.593046, -122.300124)
    assert sat_latlon("satellite_41.8528178_-87.6296328.png") == (41.8528178, -87.6296328)


def test_load_and_subset(tmp_path):
    root = make_fake_vigor(tmp_path, n=16)
    data = load_vigor(root, ["Chicago"], "same_area_balanced_test.txt")
    assert len(data.panos) == 16 and len(data.sats) == 16
    assert data.labels.shape == (16, 4)
    # fake data: pano i has satellite i as positive
    assert (data.labels[:, 0] == range(16)).all()

    small = subset(data, 4, seed=1)
    assert len(small.panos) == 4
    for pano, label in zip(small.panos, small.labels):
        positive = small.sats[label[0]]
        assert data.panos.index(pano) == data.sats.index(positive)
