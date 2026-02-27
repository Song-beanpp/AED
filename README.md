# Belgium AED Network Optimization

Spatiotemporal analysis and optimization of Automated External Defibrillator (AED) placement across Belgium. Combines deep learning forecasting with scenario-based cost-effectiveness analysis on 395,000+ emergency intervention records.

## Motivation

Cardiac arrest survival drops ~10% per minute without defibrillation. Belgium has 15,000+ public AEDs, yet **31.2% of cardiac missions occur >500 m from the nearest AED**. This project quantifies the coverage gap, forecasts demand patterns, and evaluates the cost-effectiveness of adding new devices.

## Key Findings

| Metric | Value |
|--------|-------|
| Existing AEDs | 15,227 |
| 1 km baseline coverage | 87.3% |
| 500 m baseline coverage | 68.8% |
| Coverage gain from +100 AEDs | +0.23 pp |
| Marginal cost at saturation | ~38 kEUR / 0.01 pp |

> **Core result:** The network is **macro-saturated** — adding 100 optimally-placed AEDs improves 500 m coverage by only 0.23 percentage points. Policy should prioritize **maintenance, training, and signage** over new deployments.

## Publication Figures

### Fig 1 — Coverage Gap Map
Kernel density estimate of uncovered cardiac missions (>500 m from nearest AED).

![fig1](mda_project/data/output/figures/fig1_coverage_gap_map.png)

### Fig 2 — ConvLSTM Spatial Forecast & Architecture
Deep learning prediction of daily mission density per grid cell, with model architecture and learned convolutional filters.

![fig2](mda_project/data/output/figures/fig2_convlstm_full.png)

### Fig 3 — Scenario Saturation Analysis
Coverage saturation curve and marginal cost analysis across 7 deployment scenarios (S10–S100).

![fig3](mda_project/data/output/figures/fig3_scenario_saturation.png)

### Fig 4 — Province-Level Policy Gap
Per-province gap ratio showing where the highest proportion of missions exceed the 500 m threshold.

![fig4](mda_project/data/output/figures/fig4_province_gap_choropleth.png)

## Pipeline

```
01_data_inventory.ipynb          → Raw data audit & schema comparison
02_data_preprocessing.ipynb      → Data quality checks & cleaning
03_geospatial_baseline.ipynb     → Coverage analysis & province statistics
04_predictive_modeling.ipynb     → Random Forest / GBM response time prediction
05_multiobjective_optimization.ipynb → KMeans scenario-based cost-effectiveness
06_spatiotemporal_deep_learning.ipynb → ConvLSTM grid density forecasting (PyTorch)
07_lifecycle_environmental_analysis.ipynb → 10-year CAPEX+OPEX & CO₂ lifecycle
run_all_notebooks.py             → End-to-end reproducible pipeline + publication figures
```

## Data

| File | Records | Description |
|------|---------|-------------|
| `interventions*.parquet.gzip` | 601,881 | Emergency dispatch records (3 regional files) |
| `interventions_bxl2.parquet.gzip` | 38,620 | Brussels Capital Region (separate dispatch) |
| `aed_locations.parquet.gzip` | 15,138 | Public AED registry |
| `BELGIUM_-_Provinces.geojson` | 11 | Administrative boundaries |

> Raw data files are not included due to privacy restrictions.

### Data Coverage Notes

| Province | Status | Reason |
|----------|--------|--------|
| ANT, LIM, BRW, HAI, LIE, NAM, LUX | ✅ Full data | — |
| WVL | ❌ 5 valid records | Geocoding corrupt in source (554/559 coords outside Belgium) |
| OVL | ❌ 1 record | Virtually absent from source |
| VBR | ❌ 5 records | Minimal coverage in source |
| BXL | ❌ Separate file | Different dispatch system, no province column |

## Quick Start

```bash
pip install -r requirements.txt

# Run the full pipeline (NB 01-07 + publication figures)
python run_all_notebooks.py

# Or explore notebooks individually
jupyter notebook
```

## Requirements

- Python 3.9+
- PyTorch (MPS/CUDA optional)
- See `requirements.txt`

## License

MIT
