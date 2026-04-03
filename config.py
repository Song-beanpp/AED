"""
config.py
=========
Centralized configuration for the Belgium AED Optimization project.
All paths, hyperparameters, and constants are managed here to eliminate
hardcoded values across notebooks and scripts.
"""
from pathlib import Path
import os

# ── Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR      = PROJECT_ROOT / 'mda_project' / 'data' / 'raw'
V3_DIR       = PROJECT_ROOT / 'mda_project' / 'data' / 'processed_v3'
OUTPUT_DIR   = PROJECT_ROOT / 'mda_project' / 'data' / 'output'
FIGURES_DIR  = OUTPUT_DIR / 'figures'

# ── Random Seeds ───────────────────────────────────────────────────
SEED = int(os.environ.get('AED_SEED', 42))

# ── Geospatial Constants ───────────────────────────────────────────
EARTH_RADIUS_KM = 6371.0088
BELGIUM_LAT_RANGE = (49.0, 52.0)
BELGIUM_LON_RANGE = (2.0, 7.0)
COVERAGE_THRESHOLDS_KM = [0.2, 0.5, 1.0]

# ── ConvLSTM Hyperparameters ──────────────────────────────────────
CONVLSTM_SEQ_LEN     = 5
CONVLSTM_EPOCHS      = 30
CONVLSTM_LR          = 0.003
CONVLSTM_BATCH_SIZE  = 8
CONVLSTM_DROPOUT     = 0.3
GRID_LAT_BINS        = 50
GRID_LON_BINS        = 50

# ── ML Model Parameters ──────────────────────────────────────────
RF_N_ESTIMATORS  = 100
RF_MIN_LEAF      = 5
GBM_MAX_DEPTH    = 6
CV_N_SPLITS      = 5
ML_SAMPLE_SIZE   = 50_000   # subsample for tractable training

# ── Scenario Analysis ────────────────────────────────────────────
SCENARIO_SIZES       = [10, 20, 30, 40, 50, 70, 100]
COST_PER_AED_EUR     = 2_000     # unit acquisition cost
ANNUAL_MAINT_EUR     = 120       # annual maintenance per unit
LIFECYCLE_YEARS      = 10

# ── Visualization ────────────────────────────────────────────────
FONT_TITLE = 15
FONT_LABEL = 13
FONT_TICK  = 11
FONT_ANNOT = 10
FIGURE_DPI = 250
