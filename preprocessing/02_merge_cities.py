"""
Stage 2 — Merge Cities
=======================
Combines the 4 cleaned CSVs from Stage 1 into one master file.

INPUT (4 cleaned files in data/processed/):
    flats_cleaned.csv               (Gurgaon)
    house_cleaned.csv               (Gurgaon)
    bangalore_flats_cleaned.csv     (Bangalore)
    bangalore_house_cleaned.csv     (Bangalore)

OUTPUT (data/processed/all_properties.csv):
    Merged file with explicit 'city' column added.

USAGE:
    python preprocessing/02_merge_cities.py
"""

import os
import pandas as pd


def merge_cities(
    gurgaon_flats:    str = 'data/processed/flats_cleaned.csv',
    gurgaon_houses:   str = 'data/processed/house_cleaned.csv',
    bangalore_flats:  str = 'data/processed/bangalore_flats_cleaned.csv',
    bangalore_houses: str = 'data/processed/bangalore_house_cleaned.csv',
    output_path:      str = 'data/processed/all_properties.csv',
) -> pd.DataFrame:
    """
    Combine all four cleaned files into one CSV with a 'city' column.

    Returns the merged DataFrame, ready for stage 3 (sector extraction).
    """
    print(f"\n[Stage 2] Merging cities into {output_path}")
    parts = []

    sources = [
        (gurgaon_flats,    'gurgaon'),
        (gurgaon_houses,   'gurgaon'),
        (bangalore_flats,  'bangalore'),
        (bangalore_houses, 'bangalore'),
    ]

    for path, city in sources:
        if os.path.isfile(path):
            df = pd.read_csv(path)
            df['city'] = city
            parts.append(df)
            print(f"  + {os.path.basename(path):35s} {len(df):>5} rows  city={city}")
        else:
            print(f"  - {os.path.basename(path):35s} not found, skipping")

    if not parts:
        raise FileNotFoundError("No input files found in data/processed/")

    df = pd.concat(parts, ignore_index=True)
    df = df.sample(frac=1, random_state=42, ignore_index=True)   # shuffle
    df.to_csv(output_path, index=False)

    print(f"\n  ✓ Combined: {df.shape} → {output_path}")
    print(f"\n  Breakdown by city × property_type:")
    print(df.groupby(['city', 'property_type']).size().to_string())

    return df


if __name__ == '__main__':
    merge_cities()
    print("\n✓ Stage 2 complete. Next: python preprocessing/03_extract_sector.py")
