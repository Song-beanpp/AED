#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

PROJECT_ROOT = Path('/Users/Zhuanz/S/已完成任务/mda 项目/Belgium-AED-Optimization')
DATA_DIR = PROJECT_ROOT / 'mda_project' / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
OUTPUT_DIR = DATA_DIR / 'output'
V3_DIR = DATA_DIR / 'processed_v3'
V3_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INTERVENTION_FILES = [
    'interventions1.parquet.gzip',
    'interventions2.parquet.gzip',
    'interventions3.parquet.gzip',
    'interventions_bxl2.parquet.gzip',
]

EARTH_RADIUS_KM = 6371.0088
BELGIUM_BBOX = {'lat_min': 49.0, 'lat_max': 52.0, 'lon_min': 2.0, 'lon_max': 7.0}


def normalize_col(col: str) -> str:
    col = col.strip().lower()
    col = re.sub(r'[^a-z0-9]+', '_', col)
    return re.sub(r'_+', '_', col).strip('_')


def parse_datetime_mixed(s: pd.Series) -> pd.Series:
    s_str = s.astype(str).str.strip()
    dt = pd.Series(pd.NaT, index=s.index, dtype='datetime64[ns]')

    # Custom EMS format: 01JUN22:00:01:34
    mask_custom = s_str.str.match(r'^\d{2}[A-Z]{3}\d{2}:\d{2}:\d{2}:\d{2}$', na=False)
    if mask_custom.any():
        dt.loc[mask_custom] = pd.to_datetime(
            s_str.loc[mask_custom],
            format='%d%b%y:%H:%M:%S',
            errors='coerce',
        )

    # ISO with milliseconds: 2022-06-01 00:17:53.888
    mask_iso_ms = (
        ~mask_custom
        & s_str.str.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+$', na=False)
    )
    if mask_iso_ms.any():
        dt.loc[mask_iso_ms] = pd.to_datetime(
            s_str.loc[mask_iso_ms],
            format='%Y-%m-%d %H:%M:%S.%f',
            errors='coerce',
        )

    # ISO without milliseconds fallback
    mask_iso = (
        ~mask_custom
        & ~mask_iso_ms
        & s_str.str.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', na=False)
    )
    if mask_iso.any():
        dt.loc[mask_iso] = pd.to_datetime(
            s_str.loc[mask_iso],
            format='%Y-%m-%d %H:%M:%S',
            errors='coerce',
        )

    return dt


def read_raw_interventions() -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for fn in INTERVENTION_FILES:
        path = RAW_DIR / fn
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_parquet(path)
        df.columns = [normalize_col(c) for c in df.columns]
        df['source_file'] = fn
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    return out


def mode_or_nan(s: pd.Series):
    s2 = s.dropna().astype(str)
    if s2.empty:
        return np.nan
    return s2.mode().iloc[0]


def bool_contains(s: pd.Series, pat: str) -> pd.Series:
    return s.astype(str).str.contains(pat, case=False, na=False)


def nearest_km(src_latlon: np.ndarray, dst_latlon: np.ndarray) -> np.ndarray:
    if src_latlon.size == 0 or dst_latlon.size == 0:
        return np.full(src_latlon.shape[0], np.nan)
    tree = BallTree(np.radians(dst_latlon), metric='haversine')
    dist, _ = tree.query(np.radians(src_latlon), k=1)
    return dist[:, 0] * EARTH_RADIUS_KM


def build_tables() -> Dict[str, pd.DataFrame]:
    raw = read_raw_interventions()

    # Standardized field mapping
    work = pd.DataFrame({
        'mission_id': raw.get('mission_id'),
        'latitude': pd.to_numeric(raw.get('latitude_intervention'), errors='coerce'),
        'longitude': pd.to_numeric(raw.get('longitude_intervention'), errors='coerce'),
        'province_raw': raw.get('province_intervention').astype(str),
        'event_type': raw.get('eventtype_trip').astype(str),
        'event_level': raw.get('eventlevel_trip').astype(str),
        'vector_type': raw.get('vector_type').astype(str),
        't0_raw': raw.get('t0').astype(str),
        't1_raw': raw.get('t1').astype(str),
        't2_raw': raw.get('t2').astype(str),
        't3_raw': raw.get('t3').astype(str),
        'source_file': raw['source_file'],
    })

    work['t0'] = parse_datetime_mixed(raw.get('t0'))
    work['t1'] = parse_datetime_mixed(raw.get('t1'))
    work['t2'] = parse_datetime_mixed(raw.get('t2'))
    work['t3'] = parse_datetime_mixed(raw.get('t3'))

    # Response-time definitions
    work['response_min_t3_t0'] = (work['t3'] - work['t0']).dt.total_seconds() / 60.0
    work['response_min_t2_t0'] = (work['t2'] - work['t0']).dt.total_seconds() / 60.0

    # Data validity flags
    work['geo_valid'] = (
        work['latitude'].between(BELGIUM_BBOX['lat_min'], BELGIUM_BBOX['lat_max'])
        & work['longitude'].between(BELGIUM_BBOX['lon_min'], BELGIUM_BBOX['lon_max'])
    )
    work['rt_t3_valid'] = work['response_min_t3_t0'].between(0.5, 240)
    work['rt_t2_valid'] = work['response_min_t2_t0'].between(0.5, 240)

    # Main response metric with fallback
    work['response_min'] = np.where(work['rt_t3_valid'], work['response_min_t3_t0'], np.nan)
    work.loc[work['response_min'].isna() & work['rt_t2_valid'], 'response_min'] = work.loc[
        work['response_min'].isna() & work['rt_t2_valid'], 'response_min_t2_t0'
    ]

    # Event flags
    work['is_cardiac'] = bool_contains(work['event_type'], r'cardiac|hart|p003|arrest')
    work['is_aed_vector'] = bool_contains(work['vector_type'], r'^aed$|aed')
    work['is_ambulance_vector'] = bool_contains(work['vector_type'], r'ambu')
    work['is_mug_vector'] = bool_contains(work['vector_type'], r'\bmug\b')
    work['is_pit_vector'] = bool_contains(work['vector_type'], r'\bpit\b')

    # Keep rows usable for analytical dispatch-level tasks
    dispatch = work.copy()
    dispatch['dispatch_valid'] = dispatch['geo_valid'] & dispatch['response_min'].notna() & dispatch['mission_id'].notna()

    # Mission-level table (one row per mission)
    mission = (
        dispatch[dispatch['mission_id'].notna()]
        .groupby('mission_id', as_index=False)
        .agg(
            latitude=('latitude', 'median'),
            longitude=('longitude', 'median'),
            t0=('t0', 'min'),
            t3=('t3', 'min'),
            province=('province_raw', mode_or_nan),
            event_type=('event_type', mode_or_nan),
            event_level=('event_level', mode_or_nan),
            response_min=('response_min', 'min'),
            n_dispatch=('mission_id', 'size'),
            n_vector_types=('vector_type', lambda s: s.astype(str).nunique()),
            has_aed=('is_aed_vector', 'max'),
            has_ambulance=('is_ambulance_vector', 'max'),
            has_mug=('is_mug_vector', 'max'),
            has_pit=('is_pit_vector', 'max'),
        )
    )

    mission['geo_valid'] = (
        mission['latitude'].between(BELGIUM_BBOX['lat_min'], BELGIUM_BBOX['lat_max'])
        & mission['longitude'].between(BELGIUM_BBOX['lon_min'], BELGIUM_BBOX['lon_max'])
    )
    mission['response_valid'] = mission['response_min'].between(0.5, 240)
    mission = mission[mission['geo_valid'] & mission['response_valid']].copy()

    # AED coordinates: use processed geocoded table and strict bounding
    aed = pd.read_csv(PROCESSED_DIR / 'aed_total_coordinates.csv')
    aed['latitude'] = pd.to_numeric(aed['latitude'], errors='coerce')
    aed['longitude'] = pd.to_numeric(aed['longitude'], errors='coerce')
    aed = aed[
        aed['latitude'].between(BELGIUM_BBOX['lat_min'], BELGIUM_BBOX['lat_max'])
        & aed['longitude'].between(BELGIUM_BBOX['lon_min'], BELGIUM_BBOX['lon_max'])
    ].copy()

    # Nearest AED distance for mission table
    mission['dist_to_aed_km'] = nearest_km(
        mission[['latitude', 'longitude']].to_numpy(),
        aed[['latitude', 'longitude']].to_numpy(),
    )

    return {'dispatch': dispatch, 'mission': mission, 'aed': aed}


def build_audit(dispatch: pd.DataFrame, mission: pd.DataFrame, aed: pd.DataFrame) -> Dict:
    report: Dict[str, object] = {}

    report['dispatch_n_total'] = int(len(dispatch))
    report['dispatch_n_valid'] = int(dispatch['dispatch_valid'].sum())
    report['dispatch_geo_invalid'] = int((~dispatch['geo_valid']).sum())
    report['dispatch_response_missing'] = int(dispatch['response_min'].isna().sum())
    report['dispatch_dup_spacetime'] = int(dispatch.duplicated(['mission_id', 'latitude', 'longitude', 't0', 't3']).sum())

    report['mission_n_total_valid'] = int(len(mission))
    report['mission_response_p50'] = float(mission['response_min'].median())
    report['mission_response_p90'] = float(mission['response_min'].quantile(0.9))
    report['mission_response_p99'] = float(mission['response_min'].quantile(0.99))

    report['aed_n_valid'] = int(len(aed))
    report['aed_bbox'] = {
        'lat_min': float(aed['latitude'].min()),
        'lat_max': float(aed['latitude'].max()),
        'lon_min': float(aed['longitude'].min()),
        'lon_max': float(aed['longitude'].max()),
    }

    report['mission_by_province_top'] = (
        mission['province'].value_counts(dropna=False).head(15).to_dict()
    )
    report['event_top'] = mission['event_type'].value_counts(dropna=False).head(15).to_dict()

    # Data quality decision log
    report['quality_decisions'] = [
        'Use mixed datetime parser: generic parser + custom EMS format %d%b%y:%H:%M:%S',
        'Use mission-level minimum valid response time across vectors as service-performance proxy',
        'Restrict geospatial records to Belgium bounding box (49-52N, 2-7E)',
        'Clip response-time validity to [0.5, 240] minutes for robust modeling baseline',
        'Use geocoded AED table from processed/aed_total_coordinates.csv because raw AED table lacks coordinates',
    ]

    return report


def save_schema_markdown() -> None:
    schema_md = PROJECT_ROOT / 'mda_project' / 'data' / 'output' / 'dataset_schema_v3.md'
    schema_md.write_text(
        """# Dataset Schema v3

## dispatch_records_v3.parquet
- `mission_id`: mission identifier (raw dispatch-level)
- `latitude`, `longitude`: intervention coordinates
- `t0`, `t1`, `t2`, `t3`: parsed dispatch timestamps
- `response_min_t3_t0`: minutes from call open (`t0`) to scene/phase marker (`t3`)
- `response_min_t2_t0`: fallback response minutes (`t2 - t0`)
- `response_min`: primary response metric with fallback logic
- `event_type`, `event_level`, `vector_type`: event and response-vector attributes
- `is_cardiac`, `is_aed_vector`, `is_ambulance_vector`, `is_mug_vector`, `is_pit_vector`: engineered flags
- `geo_valid`, `rt_t3_valid`, `rt_t2_valid`, `dispatch_valid`: quality flags
- `source_file`: raw source partition

## mission_records_v3.parquet
- One row per mission (`mission_id`)
- Coordinates are mission-level medians from dispatch rows
- `response_min` is mission-level minimum valid response time
- `n_dispatch`, `n_vector_types`: multi-resource complexity indicators
- `has_*` fields: whether mission involved each vector type
- `dist_to_aed_km`: nearest AED distance from cleaned geocoded AED set

## aed_records_v3.parquet
- Cleaned AED coordinates (from geocoded processed table)
- Strictly filtered to Belgium bbox
""",
        encoding='utf-8',
    )


def main() -> None:
    tables = build_tables()
    dispatch = tables['dispatch']
    mission = tables['mission']
    aed = tables['aed']

    # Save robust data products
    dispatch_out = V3_DIR / 'dispatch_records_v3.parquet'
    mission_out = V3_DIR / 'mission_records_v3.parquet'
    aed_out = V3_DIR / 'aed_records_v3.parquet'

    dispatch.to_parquet(dispatch_out, index=False)
    mission.to_parquet(mission_out, index=False)
    aed.to_parquet(aed_out, index=False)

    # Also export csv for portability
    dispatch.sample(min(200000, len(dispatch)), random_state=42).to_csv(V3_DIR / 'dispatch_records_v3_sample.csv', index=False)
    mission.to_csv(V3_DIR / 'mission_records_v3.csv', index=False)
    aed.to_csv(V3_DIR / 'aed_records_v3.csv', index=False)

    report = build_audit(dispatch, mission, aed)
    report_path = OUTPUT_DIR / 'raw_intervention_audit_v3.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    save_schema_markdown()

    print('Saved:')
    print('-', dispatch_out)
    print('-', mission_out)
    print('-', aed_out)
    print('-', report_path)
    print('-', OUTPUT_DIR / 'dataset_schema_v3.md')
    print('\nSummary:')
    for k in ['dispatch_n_total', 'dispatch_n_valid', 'dispatch_geo_invalid', 'dispatch_response_missing', 'mission_n_total_valid', 'mission_response_p50', 'mission_response_p90']:
        print(k, report[k])


if __name__ == '__main__':
    main()
