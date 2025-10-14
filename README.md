# Predictive Mapping of Oil Spill‑Induced Mangrove Degradation in Nigeria

> **Repository Project** – A geospatial data science workflow for monitoring, predicting, and mitigating oil‑spill‑induced mangrove degradation in the Niger Delta. This project integrates **remote sensing**, **spatiotemporal analysis**, and **machine learning** to produce actionable insights for **environmental management, restoration planning, and policy development**.

---

## 🌍 Project Overview

Oil spills are among the most severe environmental challenges facing the Niger Delta, where mangrove ecosystems play a crucial role in biodiversity, carbon storage, coastal protection, and livelihoods. Decades of oil exploration and pipeline failures have led to widespread degradation, threatening socio‑ecological systems and undermining climate resilience.

This project leverages multi‑temporal satellite data, official oil spill records, terrain models, and ecological sensitivity indices to **map past damage, quantify ongoing degradation, and predict future risk hotspots**. It combines **geospatial analysis, vegetation indices, environmental sensitivity modeling, and machine learning (GBDT/XGBoost)** into a reproducible pipeline that can support environmental agencies, NGOs, and policymakers.

---

## 🧭 Key Objectives

1. **Detect and quantify mangrove degradation** using satellite‑derived vegetation indices (NDVI, RENDVI) from 2020–2025.
2. **Integrate oil spill event data** from NOSDRA and other public records to understand spatial patterns and severity.
3. **Model sensitivity and risk** by combining elevation, proximity to spill clusters, and environmental sensitivity indices (ESI).
4. **Train predictive models** (XGBoost) to map future degradation hotspots and classify risk tiers.
5. **Deliver interpretable outputs** (SHAP feature attribution, probability maps, risk layers) to guide decision‑making.

---

## 🛰️ Methodological Framework

### 1. **Data Collection & Preprocessing**

* **Satellite Imagery:** Sentinel‑2 (10 m) and Landsat‑8 (30 m) imagery processed to surface reflectance (L2A).
* **Oil Spill Events:** NOSDRA spill records (2023–2025) parsed and geocoded.
* **Digital Elevation Model (DEM):** SRTM 30 m for elevation band classification.
* **Environmental Sensitivity Index (ESI):** NOAA/NNPC‑derived shapefiles identifying mangrove classes (e.g., 10a, 10b).
* **Baseline Mangrove Extent:** Global Mangrove Watch (GMW) 2020 dataset.

### 2. **Feature Engineering**

* Vegetation Change: ΔNDVI, ΔRENDVI between 2020 and 2024 epochs.
* Oil Spill Pressure: KDE‑derived hotspot layers, k‑means severity clusters.
* Sensitivity Layers: Elevation bands (<5 m, 5–10 m, >10 m), weighted ESI.
* Land Cover Dynamics: LULC transitions (e.g., flooded vegetation → bare ground).

### 3. **Machine Learning Modeling**

* **Algorithm:** Gradient Boosted Decision Trees (XGBoost)
* **Features:** ΔNDVI, ΔRENDVI, spill intensity, elevation band, ESI weight, LULC change.
* **Validation:** Spatially stratified 5‑fold cross‑validation.
* **Metrics:** Accuracy, Precision, Recall, F1‑score, AUC.
* **Interpretability:** SHAP values to quantify feature contributions.

### 4. **Predictive Mapping & Risk Classification**

* Probability raster outputs (0–1) converted into **risk tiers:** very low, low, moderate, high, very high.
* Hotspot overlays and vulnerability maps support **prioritization of restoration sites**.

---

## 📊 Sample Results & Insights

**Spatiotemporal Trends (2020–2025):**

* Mangrove canopy cover declined by ~14% across the study area.
* High‑severity degradation (>0.6 ΔNDVI) is concentrated near pipeline corridors and spill clusters in Bayelsa and Rivers States.
* KDE analysis shows persistent spill hotspots near Bodo Creek and Tombia.
* SHAP results indicate ΔNDVI (~0.38) and spill intensity (~0.27) as the top predictors of degradation probability.

**Model Performance:**

* Accuracy: 0.91
* Precision: 0.88
* Recall: 0.86
* F1‑score: 0.87
* AUC: 0.93

---

## 📁 Repository Structure

```
Predictive-Mangrove-Degradation/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ requirements.txt
├─ environment.yml
├─ configs/
│  ├─ study_area.geojson
│  ├─ params.yaml
│  └─ classes.json
├─ data/
│  ├─ raw/
│  ├─ interim/
│  └─ processed/
├─ notebooks/
│  ├─ 00_explore_AOI.ipynb
│  ├─ 20_LULC_RF.ipynb
│  ├─ 40_train_xgboost.ipynb
│  └─ 50_shap_interpretation.ipynb
├─ src/
│  ├─ features/
│  ├─ modeling/
│  ├─ spill/
│  └─ utils/
├─ models/
│  └─ artifacts/
└─ figures/
```

---

## 📊 Outputs

* **Degradation probability maps** (GeoTIFF)
* **Risk classification shapefiles** (vector)
* **Spill hotspot maps** (KDE & clusters)
* **Feature importance visualizations** (SHAP plots)
* **Land cover change maps** (LULC)

---

## 📚 Citation

> Amos M, Akajiaku, U.C., Eteh, D.R., et al. (2025). *Predictive Mapping of Oil Spill‑Induced Mangrove Degradation in Nigeria Using Remote Sensing and Machine Learning*. GitHub Repository: [https://github.com/Akajiaku11](https://github.com/Akajiaku11)

---

## 📜 License

MIT License – See `LICENSE` file for details.

---

## 🤝 Acknowledgements

* Nigerian Oil Spill Detection and Response Agency (NOSDRA)
* Global Mangrove Watch (GMW)
* Copernicus Open Access Hub (ESA)
* USGS EarthExplorer
* NOAA ESI Shapefiles

---

## 🔧 Next Steps

* [ ] Add more spill years (back to 2010) for long‑term trend analysis.
* [ ] Integrate SAR data for improved wetland detection.
* [ ] Build a dashboard for interactive risk map visualization.

---

**Contact:** [@Akajiaku11](https://github.com/Akajiaku11) – Contributions, feedback, and collaborations are welcome!
