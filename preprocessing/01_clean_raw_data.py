"""
Stage 1 — Initial Data Cleaning
================================
Cleans raw scraped CSVs from 99acres for both cities.

INPUT (4 files, in data/raw/):
    flats.csv               — Gurgaon flats
    houses.csv              — Gurgaon houses
    bangalore_flats.csv     — Bangalore flats
    bangalore_houses.csv    — Bangalore houses

OUTPUT (4 cleaned files, in data/processed/):
    flats_cleaned.csv
    house_cleaned.csv
    bangalore_flats_cleaned.csv
    bangalore_house_cleaned.csv

WHAT IT DOES:
  - Parses price strings (multiple formats: "1.7 Crore", "85 Lac", "₹0.94 - 1.34 Cr")
  - Parses bedRoom strings ("3 Bedrooms", "2-3 BHK")
  - Parses area (areaWithType, super built-up, carpet, ranges)
  - Recovers data from descriptions for project-page rows (~75% of Bangalore)
  - Standardises society names, balcony, additionalRoom
  - Drops empty rows + duplicates
  - Adds property_type column (flat/house)

USAGE:
    python preprocessing/01_clean_raw_data.py
"""

import os
import re
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# UNIVERSAL FIELD PARSERS — handle every format we found in
# the scraped data (Gurgaon + Bangalore, flats + houses)
# ─────────────────────────────────────────────────────────────

def parse_price_to_cr(s) -> float:
    """
    Convert any price string to a float in Crore.

    Handles:  '1.7 Crore' → 1.7
              '85 Lac'    → 0.85
              '₹85 Lac'   → 0.85
              '₹ 0.94 - 1.34 Cr'   → 1.14   (midpoint)
              'Price on Request'    → NaN
    """
    if pd.isna(s) or not str(s).strip():
        return np.nan
    s = str(s).strip()
    if 'Price on Request' in s or 'on Request' in s:
        return np.nan
    s = s.replace('₹', '').strip()

    # Range: "0.94 - 1.34 Cr" or "59 - 99 Lac"
    m = re.search(r'([\d.]+)\s*[-–]\s*([\d.]+)\s*(Cr|Crore|Lac|L)\b',
                  s, re.IGNORECASE)
    if m:
        lo, hi, unit = float(m.group(1)), float(m.group(2)), m.group(3).upper()
        midpoint = (lo + hi) / 2
        return round(midpoint / 100 if unit.startswith('L') else midpoint, 4)

    # Single value: "1.7 Crore", "85 Lac"
    m = re.search(r'([\d.]+)\s*(Crore|Cr|Lac|L)\b', s, re.IGNORECASE)
    if m:
        val, unit = float(m.group(1)), m.group(2).upper()
        return round(val / 100 if unit.startswith('L') else val, 4)
    return np.nan


def parse_bedroom(s) -> int:
    """
    Convert bedRoom string to int.

    Handles:  '3 Bedrooms' → 3
              '2-3 BHK'    → 3   (upper bound — typical demand)
              '1 Bedroom'  → 1
    """
    if pd.isna(s) or not str(s).strip():
        return np.nan
    s = str(s).strip()
    m = re.search(r'(\d+)\s*[-–]\s*(\d+)\s*BHK', s, re.IGNORECASE)
    if m:
        return int(m.group(2))
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else np.nan


def parse_area_to_sqft(s) -> float:
    """
    Extract sqft area from any 'areaWithType' format.

    Handles:  '1,164 - 1,758 sqft'                          → 1461 (midpoint)
              '1,114 sqft'                                  → 1114
              'Super Built-up area 1130(104.98 sq.m.)'      → 1130
              'Carpet area: 1250'                           → 1250
    """
    if pd.isna(s) or not str(s).strip():
        return np.nan
    s = str(s).strip()

    m = re.search(r'([\d,]+)\s*[-–]\s*([\d,]+)\s*sqft', s, re.IGNORECASE)
    if m:
        lo = float(m.group(1).replace(',', ''))
        hi = float(m.group(2).replace(',', ''))
        return round((lo + hi) / 2)

    m = re.search(r'(?:Super Built-?up area|Built\s*Up\s*area|Carpet\s*area)'
                  r'\s*:?\s*([\d,]+)', s, re.IGNORECASE)
    if m:
        return float(m.group(1).replace(',', ''))

    m = re.search(r'([\d,]+)\s*sqft', s, re.IGNORECASE)
    if m:
        return float(m.group(1).replace(',', ''))

    m = re.search(r'([\d,]+)', s)
    if m:
        v = float(m.group(1).replace(',', ''))
        return v if v > 100 else np.nan
    return np.nan


def parse_int_field(s) -> int:
    """For bathroom — extract integer."""
    if pd.isna(s) or not str(s).strip():
        return np.nan
    m = re.search(r'(\d+)', str(s))
    return int(m.group(1)) if m else np.nan


def parse_balcony(s) -> str:
    """Balcony: '0', '1', '2', '3', '3+'."""
    if pd.isna(s) or not str(s).strip():
        return ''
    s = str(s).strip()
    if 'No' in s:
        return '0'
    m = re.search(r'(\d+\+?)', s)
    return m.group(1) if m else s.split()[0]


def parse_floor(s) -> float:
    """Numeric floor; Ground=0, Basement=-1, Lower=0."""
    if pd.isna(s) or not str(s).strip():
        return np.nan
    s = str(s).strip()
    s = re.sub(r'\bGround\b', '0', s)
    s = re.sub(r'\bBasement\b', '-1', s)
    s = re.sub(r'\bLower\b', '0', s)
    m = re.search(r'(-?\d+)', s)
    return float(m.group(1)) if m else np.nan


def clean_society(s) -> str:
    """Strip star ratings, lowercase."""
    if pd.isna(s):
        return ''
    s = re.sub(r'\d+(\.\d+)?\s?★', '', str(s)).strip().lower()
    return s if s and s != 'nan' else ''


# ─────────────────────────────────────────────────────────────
# DESCRIPTION RECOVERY
# Many Bangalore listings are project-pages where individual
# fields are blank but the description has all the data.
# ─────────────────────────────────────────────────────────────

def parse_description(desc: str) -> dict:
    """Extract BHK / price range / area range from description text."""
    if not isinstance(desc, str) or not desc.strip():
        return {}
    out = {}

    m = re.search(r'(\d(?:\s*,\s*\d)*)\s*BHK', desc, re.IGNORECASE)
    if m:
        bhks = [int(x.strip()) for x in m.group(1).split(',')]
        out['bedRoom_min'] = min(bhks)
        out['bedRoom_max'] = max(bhks)

    m = re.search(
        r'Rs\.?\s*([\d.]+)\s*(L|Lac|Cr)?\s*[-–]\s*([\d.]+)\s*(L|Lac|Cr)',
        desc, re.IGNORECASE)
    if m:
        lo, lo_u, hi, hi_u = m.groups()
        lo_u = (lo_u or hi_u).upper()
        hi_u = hi_u.upper()
        out['price_min_cr'] = float(lo) / (100 if lo_u.startswith('L') else 1)
        out['price_max_cr'] = float(hi) / (100 if hi_u.startswith('L') else 1)

    m = re.search(r'([\d,]+)\s*[-–]\s*([\d,]+)\s*(sqft|sq\.\s*ft\.?)',
                  desc, re.IGNORECASE)
    if m:
        out['area_min_sqft'] = int(m.group(1).replace(',', ''))
        out['area_max_sqft'] = int(m.group(2).replace(',', ''))

    return out


def recover_from_description(row: dict) -> dict:
    """Fill missing price/bedRoom/areaWithType from description."""
    desc = row.get('description', '')
    if not isinstance(desc, str) or not desc.strip():
        return row

    parsed = parse_description(desc)

    if (pd.isna(row.get('bedRoom')) or
        not str(row.get('bedRoom', '')).strip()) and 'bedRoom_max' in parsed:
        row['bedRoom'] = (
            f"{parsed['bedRoom_min']}-{parsed['bedRoom_max']} BHK"
            if parsed['bedRoom_min'] != parsed['bedRoom_max']
            else f"{parsed['bedRoom_min']} Bedrooms"
        )
    if (pd.isna(row.get('price')) or
        not str(row.get('price', '')).strip()) and 'price_max_cr' in parsed:
        row['price'] = (f"₹ {parsed['price_min_cr']} - "
                        f"{parsed['price_max_cr']} Cr")
    if (pd.isna(row.get('areaWithType')) or
        not str(row.get('areaWithType', '')).strip()) and 'area_max_sqft' in parsed:
        row['areaWithType'] = (f"{parsed['area_min_sqft']:,} - "
                               f"{parsed['area_max_sqft']:,} sqft")
    return row


# ─────────────────────────────────────────────────────────────
# UNIVERSAL CLEANER — works for all 4 raw sources
# ─────────────────────────────────────────────────────────────

TARGET_COLS = [
    'property_name', 'property_type', 'society',
    'price', 'price_per_sqft', 'area', 'areaWithType',
    'bedRoom', 'bathroom', 'balcony', 'additionalRoom',
    'address', 'floorNum', 'facing', 'agePossession',
    'nearbyLocations', 'description', 'furnishDetails',
    'features', 'rating',
]


def clean(input_path: str, output_path: str, property_type: str) -> pd.DataFrame:
    """
    Clean a raw scraped CSV. Works for both flats and houses,
    both Gurgaon and Bangalore — same logic, different inputs.

    Args:
        input_path:    raw CSV path
        output_path:   cleaned CSV path
        property_type: 'flat' or 'house'
    """
    print(f"\n[Stage 1] Cleaning {input_path} → {output_path}")
    df = pd.read_csv(input_path, dtype=str).fillna('')
    print(f"  Loaded {len(df)} rows")

    # Description recovery for project-page rows
    df = pd.DataFrame([recover_from_description(r) for r in df.to_dict('records')])

    # Drop empty rows
    n0 = len(df)
    df = df[~((df['price'].str.strip() == '') &
              (df['bedRoom'].str.strip() == '') &
              (df['areaWithType'].str.strip() == ''))]
    print(f"  Dropped {n0 - len(df)} fully-empty rows")

    # Dedup
    if 'property_id' in df.columns:
        n0 = len(df)
        df = df.drop_duplicates(subset=['property_id'])
        print(f"  Dropped {n0 - len(df)} duplicate property_ids")

    df = df.drop(columns=['link', 'property_id'], errors='ignore')

    # Field parsing
    df['society']   = df['society'].apply(clean_society)
    if property_type == 'house':
        df['society'] = df['society'].replace('', 'independent')
    df['price']     = df['price'].apply(parse_price_to_cr)
    df['bedRoom']   = df['bedRoom'].apply(parse_bedroom)
    df['bathroom']  = df['bathroom'].apply(parse_int_field)
    df['balcony']   = df['balcony'].apply(parse_balcony)
    df['facing']    = df['facing'].replace('', 'NA').fillna('NA')
    df['additionalRoom'] = (df['additionalRoom']
                            .replace('', 'not available')
                            .str.lower())

    # Houses: 'rate' → price_per_sqft, 'noOfFloor' → floorNum
    def _parse_psqft(x):
        s = str(x).strip()
        if not s or not re.search(r'\d', s):
            return np.nan
        # Strip everything except digits and a single decimal point
        clean = re.sub(r'[^\d.]', '', s)
        # Collapse multiple dots: "5000.." → "5000."
        clean = re.sub(r'\.+', '.', clean).rstrip('.')
        try:
            return float(clean) if clean else np.nan
        except (ValueError, TypeError):
            return np.nan

    if property_type == 'house':
        if 'rate' in df.columns:
            df['price_per_sqft'] = df['rate'].apply(_parse_psqft)
        if 'noOfFloor' in df.columns:
            df['floorNum'] = df['noOfFloor'].apply(parse_floor)
        else:
            df['floorNum'] = np.nan
    else:
        # Flats: 'area' column already holds price/sqft string
        if 'area' in df.columns:
            df['price_per_sqft'] = df['area'].apply(_parse_psqft)
        df['floorNum'] = df.get('floorNum', '').apply(parse_floor)

    df['area'] = df['areaWithType'].apply(parse_area_to_sqft)

    # Fallback: derive area from price + price_per_sqft if still missing
    fb = df['area'].isna() & df['price'].notna() & df['price_per_sqft'].notna()
    df.loc[fb, 'area'] = round(
        (df.loc[fb, 'price'] * 10_000_000) / df.loc[fb, 'price_per_sqft'])

    # Drop rows missing core fields
    n0 = len(df)
    df = df.dropna(subset=['price', 'bedRoom', 'area'])
    print(f"  Dropped {n0 - len(df)} rows missing core fields")
    df['bedRoom'] = df['bedRoom'].astype(int)

    # Bathroom imputation
    if df['bathroom'].notna().sum() > 0:
        df['bathroom'] = df['bathroom'].fillna(
            int(df['bathroom'].median())).astype(int)
    else:
        df['bathroom'] = 0

    # Insert property_type
    if 'property_type' not in df.columns:
        df.insert(1, 'property_type', property_type)

    # Reorder to TARGET_COLS
    for c in TARGET_COLS:
        if c not in df.columns:
            df[c] = ''
    df = df[TARGET_COLS]

    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved {len(df)} rows → {output_path}")
    return df


# ─────────────────────────────────────────────────────────────
# RUN ALL 4 SOURCES
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    raw_dir   = 'data/raw'
    out_dir   = 'data/processed'
    os.makedirs(out_dir, exist_ok=True)

    sources = [
        ('flats.csv',              'flats_cleaned.csv',              'flat'),
        ('houses.csv',             'house_cleaned.csv',              'house'),
        ('bangalore_flats.csv',    'bangalore_flats_cleaned.csv',    'flat'),
        ('bangalore_houses.csv',   'bangalore_house_cleaned.csv',    'house'),
    ]

    for raw_name, out_name, ptype in sources:
        raw_path = os.path.join(raw_dir, raw_name)
        out_path = os.path.join(out_dir, out_name)
        if not os.path.isfile(raw_path):
            print(f"  ⚠  {raw_path} not found — skipping")
            continue
        clean(raw_path, out_path, ptype)

    print("\n✓ Stage 1 complete. Next: python preprocessing/02_merge_cities.py")
