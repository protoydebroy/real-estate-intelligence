"""
Run the full preprocessing pipeline end-to-end.

Runs all 7 stages sequentially:

  Stage 1  → Clean 4 raw CSVs
  Stage 2  → Merge cities (add 'city' column)
  Stage 3  → Extract sector
  Stage 4  → Feature engineering
  Stage 5  → Outlier treatment
  Stage 6  → Missing value imputation
  Stage 7  → Feature selection (final 14-column dataset)

INPUT:   data/raw/{flats,houses,bangalore_flats,bangalore_houses}.csv
OUTPUT:  data/processed/all_final_v2.csv  (model-ready)

USAGE:
    python preprocessing/run_all.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Make sibling files importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the 7 stages
from importlib import import_module

stage_01 = import_module('01_clean_raw_data')
stage_02 = import_module('02_merge_cities')
stage_03 = import_module('03_extract_sector')
stage_04 = import_module('04_feature_engineering')
stage_05 = import_module('05_outlier_treatment')
stage_06 = import_module('06_impute_missing')
stage_07 = import_module('07_feature_selection')


def run_full_pipeline():
    print("=" * 60)
    print("  REAL ESTATE INTELLIGENCE — Preprocessing Pipeline")
    print("=" * 60)

    # ── Stage 1: clean each raw CSV
    raw_dir = 'data/raw'
    out_dir = 'data/processed'
    os.makedirs(out_dir, exist_ok=True)

    sources = [
        ('flats.csv',              'flats_cleaned.csv',              'flat'),
        ('houses.csv',             'house_cleaned.csv',              'house'),
        ('bangalore_flats.csv',    'bangalore_flats_cleaned.csv',    'flat'),
        ('bangalore_houses.csv',   'bangalore_house_cleaned.csv',    'house'),
    ]
    for raw, clean, ptype in sources:
        raw_path = os.path.join(raw_dir, raw)
        if os.path.isfile(raw_path):
            stage_01.clean(raw_path, os.path.join(out_dir, clean), ptype)

    # ── Stages 2-7
    stage_02.merge_cities()
    stage_03.extract_sector()
    stage_04.feature_engineering()
    stage_05.treat_outliers()
    stage_06.impute_missing()
    stage_07.feature_selection()

    print("\n" + "=" * 60)
    print("  ✓ Pipeline complete!")
    print("  Output: data/processed/all_final_v2.csv")
    print("  Next step: python model/train_model.py")
    print("=" * 60)


if __name__ == '__main__':
    run_full_pipeline()
