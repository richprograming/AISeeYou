# AISeeYou

Cross-view geolocalization: given a street-level photo, retrieve where it was taken by
matching it against aerial imagery. Work in progress.

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Data

VIGOR is research-only and cannot be redistributed. Request access via the form linked in
https://github.com/Jeff-Zilence/VIGOR/blob/main/data/DATASET.md and extract to:

```
data/VIGOR/
  Chicago/  NewYork/  SanFrancisco/  Seattle/
    panorama/    <panoid>,<lat>,<lon>,.jpg
    satellite/   satellite_<lat>_<lon>.png
  splits/
    Chicago/ ...  satellite_list.txt, same_area_balanced_{train,test}.txt, pano_label_balanced.txt
```

## Baseline

```
python -m scripts.make_fake_vigor data/fake_vigor        # pipeline check without real data
python -m scripts.baseline --root data/fake_vigor --max-queries 64
python -m scripts.baseline                                # real VIGOR, see configs/baseline.yaml
```
