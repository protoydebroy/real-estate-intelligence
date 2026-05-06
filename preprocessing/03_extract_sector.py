"""
Stage 3 — Sector Extraction
============================
Extracts the location ("sector") from the property_name field.

INPUT:   data/processed/all_properties.csv
OUTPUT:  data/processed/all_v1.csv (5,377 × 18)

WHAT IT DOES:
  - Pulls the location string after "in" in property_name
    "3 BHK Flat in Sector 57, Gurgaon"  → "sector 57"
    "4 BHK Flat in Whitefield, Bangalore" → "whitefield, bangalore"
  - Normalises ~50 known Gurgaon sub-area aliases
    "dlf phase 1"        → "sector 26"
    "sushant lok phase 3" → "sector 57"
  - Applies manual index-level fixes for ~10 mis-extracted rows
  - Drops sectors with fewer than 3 listings (not statistically representative)
  - Drops: property_name, address, description, rating

USAGE:
    python preprocessing/03_extract_sector.py
"""

import pandas as pd


# ─────────────────────────────────────────────────────────────
# GURGAON SECTOR ALIAS MAP — built from EDA of property names
# ─────────────────────────────────────────────────────────────

SECTOR_MAP = {
    'dharam colony': 'sector 12', 'krishna colony': 'sector 7',
    'suncity': 'sector 54', 'prem nagar': 'sector 13',
    'mg road': 'sector 28', 'gandhi nagar': 'sector 28',
    'laxmi garden': 'sector 11', 'shakti nagar': 'sector 11',
    'baldev nagar': 'sector 7', 'shivpuri': 'sector 7',
    'garhi harsaru': 'sector 17', 'imt manesar': 'manesar',
    'adarsh nagar': 'sector 12', 'shivaji nagar': 'sector 11',
    'bhim nagar': 'sector 6', 'madanpuri': 'sector 7',
    'saraswati vihar': 'sector 28', 'arjun nagar': 'sector 8',
    'ravi nagar': 'sector 9', 'vishnu garden': 'sector 105',
    'bhondsi': 'sector 11', 'surya vihar': 'sector 21',
    'devilal colony': 'sector 9', 'valley view estate': 'gwal pahari',
    'mehrauli  road': 'sector 14', 'jyoti park': 'sector 7',
    'ansal plaza': 'sector 23', 'dayanand colony': 'sector 6',
    'sushant lok phase 2': 'sector 55', 'chakkarpur': 'sector 28',
    'greenwood city': 'sector 45', 'subhash nagar': 'sector 12',
    'sohna road road': 'sohna road', 'malibu town': 'sector 47',
    'surat nagar 1': 'sector 104', 'new colony': 'sector 7',
    'mianwali colony': 'sector 12', 'jacobpura': 'sector 12',
    'rajiv nagar': 'sector 13', 'ashok vihar': 'sector 3',
    'dlf phase 1': 'sector 26', 'nirvana country': 'sector 50',
    'palam vihar': 'sector 2', 'dlf phase 2': 'sector 25',
    'sushant lok phase 1': 'sector 43', 'laxman vihar': 'sector 4',
    'dlf phase 4': 'sector 28', 'dlf phase 3': 'sector 24',
    'sushant lok phase 3': 'sector 57', 'dlf phase 5': 'sector 43',
    'rajendra park': 'sector 105', 'uppals southend': 'sector 49',
    'sohna': 'sohna road', 'ashok vihar phase 3 extension': 'sector 5',
    'south city 1': 'sector 41', 'ashok vihar phase 2': 'sector 5',
    'sector 95a': 'sector 95', 'sector 23a': 'sector 23',
    'sector 12a': 'sector 12', 'sector 3a': 'sector 3',
    'sector 110 a': 'sector 110', 'patel nagar': 'sector 15',
    'a block sector 43': 'sector 43', 'maruti kunj': 'sector 12',
    'b block sector 43': 'sector 43', 'sector-33 sohna road': 'sector 33',
    'sector 1 manesar': 'manesar', 'sector 4 phase 2': 'sector 4',
    'sector 1a manesar': 'manesar', 'c block sector 43': 'sector 43',
    'sector 89 a': 'sector 89', 'sector 2 extension': 'sector 2',
    'sector 36 sohna road': 'sector 36',
}


def extract_sector(
    input_path:  str = 'data/processed/all_properties.csv',
    output_path: str = 'data/processed/all_v1.csv',
) -> pd.DataFrame:
    """
    Extract sector from property_name and normalise.
    Works for both Gurgaon (with sector aliases) and Bangalore
    (neighborhoods are kept as-is).
    """
    print(f"\n[Stage 3] Extracting sector → {output_path}")
    df = pd.read_csv(input_path)
    print(f"  Loaded: {df.shape}")

    # Extract sector from property_name (text after "in")
    df.insert(3, 'sector',
        df['property_name']
          .str.split('in').str.get(1)
          .str.replace('Gurgaon', '', regex=False)
          .str.strip()
          .str.lower()
    )

    # Apply Gurgaon alias mapping
    for old, new in SECTOR_MAP.items():
        df['sector'] = df['sector'].str.replace(old, new, regex=False)

    # Additional cleanups found via EDA
    df['sector'] = df['sector'].str.replace('sector 95a', 'sector 95', regex=False)
    df['sector'] = df['sector'].str.replace('sector 23a', 'sector 23', regex=False)
    df['sector'] = df['sector'].str.replace('new sector 2', 'sector 110', regex=False)

    # Drop sectors with fewer than 3 listings (not representative)
    counts = df['sector'].value_counts()
    n0 = len(df)
    df = df[df['sector'].isin(counts[counts >= 3].index)].copy()
    print(f"  Dropped {n0 - len(df)} rows in rare sectors (<3 listings)")

    # Drop columns no longer needed for modelling
    df.drop(columns=['property_name', 'address', 'description', 'rating'],
            inplace=True, errors='ignore')

    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved → {output_path}  shape={df.shape}")
    print(f"\n  Sectors: {df['sector'].nunique()} unique")
    print(f"    Gurgaon-style:   {df[df['city']=='gurgaon']['sector'].nunique()}")
    print(f"    Bangalore-style: {df[df['city']=='bangalore']['sector'].nunique()}")
    return df


if __name__ == '__main__':
    extract_sector()
    print("\n✓ Stage 3 complete. "
          "Next: python preprocessing/04_feature_engineering.py")
