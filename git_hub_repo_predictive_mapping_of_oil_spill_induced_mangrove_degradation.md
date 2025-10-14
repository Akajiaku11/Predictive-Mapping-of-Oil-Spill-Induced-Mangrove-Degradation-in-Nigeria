# Predictive Mapping of Oil Spill‑Induced Mangrove Degradation in Nigeria

> **Repository template** for remote sensing + machine learning workflow to map, monitor, and predict oil‑spill driven mangrove degradation in the Niger Delta (Nigeria). Designed for full reproducibility and easy extension to other coastal regions.

---

## 📁 Repository Structure

```
Predictive-Mangrove-Degradation/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ .gitignore
├─ requirements.txt
├─ environment.yml
├─ Makefile
├─ dvc.yaml                 # optional if you use DVC
├─ pyproject.toml           # optional; for packaging if needed
├─ .pre-commit-config.yaml
├─ .github/
│  └─ workflows/
│     └─ ci.yml             # lint + tests
├─ configs/
│  ├─ study_area.geojson    # AOI polygon (placeholder)
│  ├─ params.yaml           # all hyperparameters & data paths
│  └─ classes.json          # LULC class map
├─ data/
│  ├─ raw/                  # (gitignored) raw downloads
│  ├─ interim/              # (gitignored) cleaned/intermediate
│  └─ processed/            # (gitignored) features/tiles ready for ML
├─ docs/
│  ├─ methodology.md
│  ├─ data_sources.md
│  ├─ model_report.md
│  └─ governance.md
├─ notebooks/
│  ├─ 00_explore_AOI.ipynb
│  ├─ 10_build_indices.ipynb
│  ├─ 20_LULC_RF.ipynb
│  ├─ 30_spill_hotspots.ipynb
│  ├─ 40_train_xgboost.ipynb
│  └─ 50_shap_interpretation.ipynb
├─ src/
│  ├─ __init__.py
│  ├─ utils/
│  │  ├─ io.py
│  │  ├─ geoutils.py
│  │  └─ viz.py
│  ├─ data/
│  │  ├─ download_sat.py
│  │  ├─ download_spills.py
│  │  ├─ build_dem.py
│  │  └─ tiles.py
│  ├─ features/
│  │  ├─ indices.py         # NDVI/RENDVI/ΔNDVI
│  │  ├─ lulc.py            # RF classification
│  │  └─ sensitivity.py     # ESI + elevation features
│  ├─ modeling/
│  │  ├─ dataset.py         # tabular feature assembly
│  │  ├─ train_gbdt.py      # XGBoost training + CV + metrics
│  │  ├─ predict.py
│  │  └─ shap_report.py
│  └─ spill/
│     ├─ kde.py             # kernel density for hotspots
│     └─ clusters.py        # k‑means severity clusters
├─ models/
│  ├─ artifacts/            # (gitignored) trained models
│  └─ reports/              # auto‑generated metrics/plots
└─ figures/                 # key PNG/SVG figures for README & papers
```

> **Tip:** clone as a template, then replace AOI and parameters in `configs/params.yaml`.

---

## 🚀 Quickstart

### 1) Create the environment

```bash
# conda (recommended)
conda env create -f environment.yml
conda activate mangrove-ml

# OR pip
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pre-commit install
```

### 2) Configure the study area & parameters

- Put your Area of Interest polygon in `configs/study_area.geojson` (WGS84).
- Edit `configs/params.yaml` to set:
  - time windows (e.g., 2020 vs 2024),
  - sensors (Sentinel‑2, Landsat‑8),
  - DEM source (SRTM 30 m),
  - NOSDRA oil‑spill API query filters,
  - ML hyperparameters (XGBoost/GBDT),
  - output tiling size & CRS (UTM 32N for Rivers State).

### 3) Pull data (satellite, DEM, spills)

```bash
# sentinel-2 / landsat-8 / SRTM using open APIs or GEE exports
python -m src.data.download_sat --cfg configs/params.yaml

# NOSDRA spills (2023–2025) JSON → GeoPackage
python -m src.data.download_spills --cfg configs/params.yaml

# Build DEM stack
python -m src.data.build_dem --cfg configs/params.yaml
```

### 4) Build features

```bash
# NDVI, RENDVI, ΔNDVI, ΔRENDVI rasters
python -m src.features.indices --cfg configs/params.yaml

# LULC (Random Forest) and change detection
python -m src.features.lulc --cfg configs/params.yaml

# KDE spill intensity, severity clusters, ESI weights, elevation bands
python -m src.spill.kde --cfg configs/params.yaml
python -m src.spill.clusters --cfg configs/params.yaml
python -m src.features.sensitivity --cfg configs/params.yaml
```

### 5) Train model & explain

```bash
# Assemble tabular dataset from raster stacks + vector overlays
python -m src.modeling.dataset --cfg configs/params.yaml

# Train Gradient Boosted Decision Trees (XGBoost)
python -m src.modeling.train_gbdt --cfg configs/params.yaml

# Generate probability maps + class maps (risk tiers)
python -m src.modeling.predict --cfg configs/params.yaml

# SHAP feature attribution report
python -m src.modeling.shap_report --cfg configs/params.yaml
```

### 6) Reproduce figures & paper tables

Jupyter notebooks in `notebooks/` regenerate all exploratory charts and publication figures. Final plots are saved to `figures/`.

---

## 🧠 Scientific Overview

**Goal.** Predict and map mangrove degradation risk driven by oil spills in the Niger Delta by integrating multi‑temporal satellite indices (NDVI, RENDVI), LULC change, DEM‑based elevation, Environmental Sensitivity Index (ESI), and oil‑spill hotspot metrics into an explainable ML model (GBDT/XGBoost).

**Core signals.**
- **ΔNDVI/ΔRENDVI:** vegetation stress and canopy loss between two epochs.
- **KDE spill intensity & k‑means severity:** spatial pressure of spills.
- **Elevation bands (e.g., <5 m):** low‑lying retention areas.
- **ESI class (e.g., 10b mangroves):** intrinsic ecological vulnerability.
- **LULC transitions:** e.g., flooded vegetation → bare/built.

**Model.** Gradient‑boosted trees with spatially stratified CV, tuned via `configs/params.yaml`. Model cards and SHAP explanations are auto‑exported to `models/reports/`.

---

## 🔧 Key Configuration (`configs/params.yaml`)

```yaml
project:
  name: predictive-mangrove-degradation
  crs: "EPSG:32632"         # UTM 32N
  tile_size: 512
  aoi_file: configs/study_area.geojson
  out_dir: data/processed

satellite:
  sensors: ["sentinel2", "landsat8"]
  s2:
    level: L2A
    date_start: "2020-01-01"
    date_end:   "2020-03-31"  # dry season example
    date_start_2: "2024-01-01"
    date_end_2:   "2024-03-31"
    cloud_pct: 20
  l8:
    date_start: "2020-01-01"
    date_end:   "2020-12-31"

dem:
  source: "SRTM_30m"
  bands: ["elevation", "slope"]
  elevation_bins: [0,5,10,100]

spills:
  source: "NOSDRA"
  years: [2023,2024,2025]
  min_volume_bbl: 1
  kde_bandwidth_m: 2500
  clusters_k: 5

esi:
  source_file: data/raw/ESI/esi_nigeria.gpkg
  class_weights:
    "10b": 3
    "10a": 2
    "9b": 2
    default: 1

lulc:
  classes: [water, trees, rangeland, flooded_veg, built, bare]
  rf_trees: 100

model:
  algo: xgboost
  test_size: 0.2
  cv_folds: 5
  params:
    n_estimators: 400
    learning_rate: 0.05
    max_depth: 6
    subsample: 0.8
    colsample_bytree: 0.8
    reg_lambda: 1.0

outputs:
  prob_threshold: 0.5
  risk_bins: [0.2, 0.4, 0.6, 0.8]
```

---

## 🧩 Example Scripts

### `src/features/indices.py`

```python
import argparse, yaml, numpy as np, rasterio
from rasterio.enums import Resampling
from pathlib import Path

# Minimal NDVI/RENDVI builder from pre-downloaded surface reflectance stacks
# Expects aligned rasters: NIR, RED, RE1 (705 nm), RE2 (740 nm)


def ndvi(nir, red):
    return (nir - red) / (nir + red + 1e-6)


def rendvi(re, nir):
    return (nir - re) / (nir + re + 1e-6)


def write_like(src_path, out_path, arr):
    with rasterio.open(src_path) as src:
        profile = src.profile
    profile.update(dtype=rasterio.float32, count=1, compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)


def main(cfg):
    in_dir = Path(cfg["project"]["out_dir"]) / "sr_stacks"
    out_dir = Path(cfg["project"]["out_dir"]) / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Example file naming convention (adjust to your pipeline)
    nir_2020 = in_dir / "nir_2020.tif"
    red_2020 = in_dir / "red_2020.tif"
    re1_2020 = in_dir / "re1_2020.tif"

    nir_2024 = in_dir / "nir_2024.tif"
    red_2024 = in_dir / "red_2024.tif"
    re1_2024 = in_dir / "re1_2024.tif"

    with rasterio.open(nir_2020) as n0, rasterio.open(red_2020) as r0:
        ndvi_2020 = ndvi(n0.read(1), r0.read(1))
    with rasterio.open(nir_2024) as n1, rasterio.open(red_2024) as r1:
        ndvi_2024 = ndvi(n1.read(1), r1.read(1))

    write_like(nir_2020, out_dir / "ndvi_2020.tif", ndvi_2020)
    write_like(nir_2024, out_dir / "ndvi_2024.tif", ndvi_2024)
    write_like(nir_2024, out_dir / "d_ndvi_2020_2024.tif", ndvi_2024 - ndvi_2020)

    with rasterio.open(re1_2020) as re0:
        rendvi_2020 = rendvi(re0.read(1), rasterio.open(nir_2020).read(1))
    with rasterio.open(re1_2024) as re1src:
        rendvi_2024 = rendvi(re1src.read(1), rasterio.open(nir_2024).read(1))

    write_like(re1_2020, out_dir / "rendvi_2020.tif", rendvi_2020)
    write_like(re1_2024, out_dir / "rendvi_2024.tif", rendvi_2024)
    write_like(re1_2024, out_dir / "d_rendvi_2020_2024.tif", rendvi_2024 - rendvi_2020)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    args = p.parse_args()
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
```

### `src/spill/kde.py`

```python
import geopandas as gpd
import numpy as np
from sklearn.neighbors import KernelDensity
from shapely.geometry import Point
from rasterio.features import rasterize
import rasterio

# Build a KDE raster of spill intensity from point events (bbl‑weighted)


def build_kde(points_gpkg, bandwidth_m, out_raster, template_raster):
    gdf = gpd.read_file(points_gpkg).to_crs("EPSG:32632")
    X = np.vstack([gdf.geometry.x.values, gdf.geometry.y.values]).T
    weights = gdf["volume_bbl"].values

    kde = KernelDensity(bandwidth=bandwidth_m, kernel="gaussian", metric="euclidean")
    kde.fit(X, sample_weight=weights)

    with rasterio.open(template_raster) as src:
        profile = src.profile
        xs = np.arange(src.bounds.left, src.bounds.right, src.res[0])
        ys = np.arange(src.bounds.bottom, src.bounds.top, src.res[1])
    xx, yy = np.meshgrid(xs, ys)
    grid = np.vstack([xx.ravel(), yy.ravel()]).T
    z = np.exp(kde.score_samples(grid)).reshape(yy.shape)

    profile.update(count=1, dtype=rasterio.float32, compress="lzw")
    with rasterio.open(out_raster, "w", **profile) as dst:
        dst.write(z.astype("float32"), 1)
```

### `src/modeling/train_gbdt.py`

```python
import argparse, yaml, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
from pathlib import Path

# Tabular dataset must include columns listed in cfg["model"]["features"]


def kfold_train(X, y, params, cv_folds):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    metrics = []
    models = []
    for tr, va in skf.split(X, y):
        dtr = xgb.DMatrix(X.iloc[tr], label=y.iloc[tr])
        dva = xgb.DMatrix(X.iloc[va], label=y.iloc[va])
        model = xgb.train(params, dtr, num_boost_round=params.get("n_estimators", 400))
        p = model.predict(dva)
        yhat = (p >= 0.5).astype(int)
        metrics.append({
            "acc": accuracy_score(y.iloc[va], yhat),
            "prec": precision_score(y.iloc[va], yhat),
            "rec": recall_score(y.iloc[va], yhat),
            "f1": f1_score(y.iloc[va], yhat),
            "auc": roc_auc_score(y.iloc[va], p),
        })
        models.append(model)
    return models, pd.DataFrame(metrics)


def main(cfg):
    df = pd.read_parquet("data/processed/training_dataset.parquet")
    features = cfg["model"].get("features", [
        "d_ndvi", "d_rendvi", "spill_kde", "elevation_bin", "esi_weight", "lulc_change"
    ])
    X = df[features]
    y = df["label_degraded"].astype(int)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": cfg["model"]["params"]["learning_rate"],
        "max_depth": cfg["model"]["params"]["max_depth"],
        "subsample": cfg["model"]["params"]["subsample"],
        "colsample_bytree": cfg["model"]["params"]["colsample_bytree"],
        "lambda": cfg["model"]["params"]["reg_lambda"],
        "n_estimators": cfg["model"]["params"]["n_estimators"],
        "verbosity": 0
    }

    models, metr = kfold_train(X, y, params, cfg["model"]["cv_folds"])
    outdir = Path("models/artifacts"); outdir.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(models):
        m.save_model(outdir / f"gbdt_fold{i}.json")
    metr.to_csv("models/reports/cv_metrics.csv", index=False)

    print(metr.describe())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    args = p.parse_args()
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
```

---

## 🧪 Testing & CI

- Run lint & tests locally:

```bash
pre-commit run --all-files
pytest -q
```

- CI (`.github/workflows/ci.yml`) runs pre‑commit and tests on pushes and PRs.

---

## 🗃️ Data Sources (placeholders)

- **Sentinel‑2 L2A** (Copernicus Open Access Hub)
- **Landsat‑8 SR** (USGS EarthExplorer)
- **NOSDRA Oil Spill Monitor** (API/CSV exports)
- **SRTM 30 m DEM** (USGS)
- **Environmental Sensitivity Index (ESI)** shapefiles (NOAA / national agency)
- **Global Mangrove Watch (GMW)** for baseline mangrove extent

> See `docs/data_sources.md` for exact links and access tips.

---

## 📊 Outputs

- **Degradation probability raster** (GeoTIFF)
- **Risk tiers** (very low → very high) vectorized for planning
- **Hotspot maps** (KDE & clusters)
- **Model card** with CV metrics + **SHAP** feature attributions
- **Change maps**: LULC, ΔNDVI/ΔRENDVI, and overlays with ESI/elevation

---

## 🧩 Reproducibility & Data Governance

- All parameters tracked in `configs/params.yaml`.
- Optional data versioning with **DVC** (see `dvc.yaml`).
- Scripts idempotent; safe to re‑run when inputs update.
- Include `docs/governance.md` to describe data licenses, consent, and ethical use.

---

## 🙌 Contributing

1. Fork & branch (`feat/…`, `fix/…`).
2. Run `pre-commit` hooks; add tests.
3. Open a PR with a clear description and screenshots of maps where relevant.

See `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md` for details.

---

## 📜 License

MIT (see `LICENSE`).

---

## ✍️ Citation

Use the generated `CITATION.cff` or cite the repository as:

> YourName et al. (2025). *Predictive Mapping of Oil Spill‑Induced Mangrove Degradation in Nigeria*. GitHub repository. https://github.com/Akajiaku11

---

## 🔖 File: `README.md`

(This README content mirrors the sections above, adapted for GitHub formatting with badges.)

---

## 🔖 File: `requirements.txt`

```
geopandas
rasterio
rioxarray
xarray
numpy
pandas
scikit-learn
xgboost
shap
pyyaml
pyproj
tqdm
matplotlib
seaborn
contextily
requests
joblib
jupyter
```

---

## 🔖 File: `environment.yml`

```yaml
name: mangrove-ml
channels: [conda-forge]
dependencies:
  - python=3.11
  - geopandas
  - rasterio
  - rioxarray
  - xarray
  - numpy
  - pandas
  - scikit-learn
  - xgboost
  - shap
  - pyyaml
  - pyproj
  - tqdm
  - matplotlib
  - contextily
  - requests
  - joblib
  - jupyterlab
  - pip
  - pip:
      - pre-commit
      - pytest
```

---

## 🔖 File: `.gitignore`

```
# Python
__pycache__/
*.pyc
.venv/

# Jupyter
.ipynb_checkpoints/

# Data & models
/data/raw/
/data/interim/
/data/processed/
/models/artifacts/
/models/reports/*.tmp

# DVC
/.dvc/
.dvc/
*.dvc

# OS
.DS_Store
Thumbs.db
```

---

## 🔖 File: `.github/workflows/ci.yml`

```yaml
name: CI
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pre-commit pytest
      - name: Pre-commit
        run: pre-commit run --all-files
      - name: Tests
        run: pytest -q
```

---

## 🔖 File: `CITATION.cff`

```yaml
cff-version: 1.2.0
title: Predictive Mapping of Oil Spill-Induced Mangrove Degradation in Nigeria
authors:
  - family-names: Eteh
    given-names: Desmond Rowland
  - family-names: Akajiaku
    given-names: Ugochukwu Charles
  - name: Contributors
version: 1.0.0
license: MIT
date-released: 2025-10-14
repository-code: https://github.com/Akajiaku11/Predictive-Mangrove-Degradation
```

---

## 🔖 File: `LICENSE`

MIT License (insert standard boilerplate with your name and year).

---

## 🔖 File: `docs/methodology.md`

- End‑to‑end narrative of preprocessing, indices, LULC, KDE/Clusters, ESI/elevation integration, ML training, spatial CV, and SHAP explanation.
- Include equations for NDVI, RENDVI, KDE, k‑means objective, and GBDT loss.

---

## 🔖 File: `docs/data_sources.md`

- How to access Copernicus (Sentinel‑2), USGS (Landsat & SRTM), NOSDRA spill data, ESI shapefiles, and GMW.
- Data licenses and acceptable‑use notes.

---

## 🔖 File: `docs/model_report.md`

- Auto‑filled by training step with CV table (accuracy/precision/recall/F1/AUC), confusion matrix, PR/ROC curves, and SHAP bar/summary plots.

---

## 🔖 File: `docs/governance.md`

- Ethical use, environmental justice considerations, and guidance for communicating uncertainty.

---

## 🔖 File: `Makefile`

```make
.PHONY: all data features train predict report

all: data features train predict report

init:
	pre-commit install

data:
	python -m src.data.download_sat --cfg configs/params.yaml
	python -m src.data.download_spills --cfg configs/params.yaml
	python -m src.data.build_dem --cfg configs/params.yaml

features:
	python -m src.features.indices --cfg configs/params.yaml
	python -m src.features.lulc --cfg configs/params.yaml
	python -m src.spill.kde --cfg configs/params.yaml
	python -m src.spill.clusters --cfg configs/params.yaml
	python -m src.features.sensitivity --cfg configs/params.yaml

train:
	python -m src.modeling.dataset --cfg configs/params.yaml
	python -m src.modeling.train_gbdt --cfg configs/params.yaml

predict:
	python -m src.modeling.predict --cfg configs/params.yaml

report:
	python -m src.modeling.shap_report --cfg configs/params.yaml
```

---

## 🧭 Next Steps for Your GitHub

1. Create a new repo under **`github.com/Akajiaku11`** named `Predictive-Mangrove-Degradation`.
2. Copy this scaffold into the repo, commit, and push.
3. Add AOI & parameters, then run `make all` to generate first results.
4. Upload key maps in `figures/` and publish a concise `docs/model_report.md`.
5. (Optional) Turn on GitHub Pages to host an interactive map or docs.

---

## ✉️ Contact & Acknowledgements

- Open an issue for questions/feature requests.
- Acknowledge Rivers State, NOSDRA, and open‑data providers in publications.

---

*This template is designed to be publication‑ready and policy‑useful (SDGs 13/14/15, blue‑carbon, environmental compliance). Replace placeholders with your study‑specific details before release.*

