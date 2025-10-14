# Predictive Mapping of Oil Spill‑Induced Mangrove Degradation in Nigeria
**Remote Sensing + Machine Learning (Python + GEE)**

This repository contains a reproducible workflow for mapping and **predicting oil spill‑induced mangrove degradation**
in the Niger Delta (Rivers State, Nigeria) using **Sentinel‑2 / Landsat 8**, **SRTM**, **NOSDRA oil‑spill records**,
and **Gradient Boosted Decision Trees (XGBoost)** with **SHAP** explainability.

> Paper context and methods adapted from the project draft provided by the authors (uploaded by the user).

## Highlights
- Compute **NDVI / Red‑Edge NDVI (RENDVI)** time series (2020 → 2024)
- Supervised **LULC classification** (Random Forest in **Google Earth Engine**) for 2020 & 2024
- **Oil spill hotspot severity** using Kernel Density Estimation (KDE) + **K‑Means**
- Feature stack: ΔNDVI, ΔRENDVI, spill density, **ESI** rank, elevation class, LULC transition
- Train **XGBoost** classifier; evaluate **Accuracy, Precision, Recall, F1, ROC‑AUC**
- **Explain predictions** with **SHAP**; export risk probability map GeoTIFF + figures

## Repository structure
```
predictive-mangrove-degradation/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ pyproject.toml
├─ requirements.txt
├─ Makefile
├─ .gitignore
├─ .github/workflows/ci.yml
├─ configs/
│  └─ rivers_state_example.yaml
├─ data/
│  ├─ raw/           # put input rasters/vectors here
│  ├─ interim/       # intermediate outputs
│  └─ processed/     # final maps & model artifacts
├─ notebooks/
│  ├─ 00_quickstart.ipynb
│  └─ 10_model_diagnostics.ipynb
├─ scripts/
│  ├─ gee_lulc_classifier.js        # RF in GEE for 2020/2024
│  └─ prepare_shapefile_grid.py
├─ src/pmd/
│  ├─ __init__.py
│  ├─ cli.py                         # command line interface
│  ├─ io.py                          # loading/saving
│  ├─ indices.py                     # NDVI/RENDVI + delta
│  ├─ geoutils.py                    # raster/vector helpers
│  ├─ spills.py                      # KDE + clustering
│  ├─ features.py                    # stack features for ML
│  ├─ model.py                       # train/eval xgboost + shap
│  ├─ visualize.py                   # plots & maps
│  └─ esi_zones.py                   # ESI handling
└─ tests/
   └─ test_imports.py
```

## Quick start
1. **Clone** this repo and create a Python env:
   ```bash
   uv venv && source .venv/bin/activate
   uv pip install -r requirements.txt
   ```
   (Or use `pip` / `conda` as you prefer.)

2. **Data placement** (`data/raw/`):
   - `sentinel_2020.tif`, `sentinel_2024.tif`   – atmospherically‑corrected surface reflectance (bands: B4, B5, B6, B8)
   - `landsat8_2020.tif`, `landsat8_2024.tif`   – surface reflectance (B4, B5)
   - `srtm_30m.tif`                             – elevation
   - `lulc_2020.tif`, `lulc_2024.tif`           – exported from **GEE** (see `scripts/gee_lulc_classifier.js`)
   - `spills_2023_2025.geojson`                 – **NOSDRA** events (point features: date, barrels, cause)
   - `esi.shp`                                  – Environmental Sensitivity Index polygons (with rank field)
   - `rivers_state_boundary.shp`                – study boundary

   All rasters should share **CRS = EPSG:32632** and **resolution = 30 m**.

3. **Configure** paths in `configs/rivers_state_example.yaml`.

4. **Run the pipeline** (end‑to‑end):
   ```bash
   python -m pmd.cli compute-indices --cfg configs/rivers_state_example.yaml
   python -m pmd.cli build-spill-features --cfg configs/rivers_state_example.yaml
   python -m pmd.cli stack-features --cfg configs/rivers_state_example.yaml
   python -m pmd.cli train-model --cfg configs/rivers_state_example.yaml
   python -m pmd.cli predict-map --cfg configs/rivers_state_example.yaml
   python -m pmd.cli explain-model --cfg configs/rivers_state_example.yaml
   ```

5. **Outputs** land here:
   - `data/processed/ndvi_2020.tif`, `ndvi_2024.tif`, `delta_ndvi.tif`
   - `data/processed/rendvi_2020.tif`, `rendvi_2024.tif`, `delta_rendvi.tif`
   - `data/interim/spill_kde.tif`, `spill_clusters.geojson`
   - `data/processed/feature_stack.parquet`
   - `data/processed/models/xgb_model.json`, `scaler.pkl`, `metrics.json`
   - `data/processed/prediction_prob.tif`, `predicted_classes.tif`
   - `figures/roc_curve.png`, `figures/shap_summary.png`, `figures/feature_importance.png`

## Google Earth Engine (LULC)
Use `scripts/gee_lulc_classifier.js` in the **GEE Code Editor** to export 2020/2024 LULC (6 classes) as GeoTIFF.

## CLI help
```bash
python -m pmd.cli --help
python -m pmd.cli compute-indices --help
```

## License
MIT (see `LICENSE`).

## Citation
If you use this repository, please cite it using `CITATION.cff` and, if applicable, cite the accompanying manuscript.

---

### Acknowledgement
Methods and study framing are aligned with the uploaded project manuscript on *Predictive Mapping of Oil Spill‑Induced Mangrove Degradation in Nigeria Using Remote Sensing and Machine Learning*. 
