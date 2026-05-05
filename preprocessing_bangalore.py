# ============================================================
# preprocessing_bangalore.py  —  Bangalore Data Preprocessing
# ============================================================
# Bangalore data has a different shape from Gurgaon:
#   - ~75% project-page rows (price/bedRoom/area as RANGES)
#   - ~25% individual-flat rows (single values)
#   - Different price formats: "Crore" (not "Cr"), "85 Lac" (no space)
#   - Address often missing, agePossession often missing
#
# This module:
#   1. Recovers data from descriptions (uses scraper.parse_description)
#   2. Parses both "single" and "range" formats for price/bedRoom/area
#   3. Converts ranges to midpoints (so ML can use them)
#   4. Produces output identical to Gurgaon's flats_cleaned.csv schema
#
# Then the main preprocessing.py pipeline can run on the cleaned files
# exactly as it does for Gurgaon — merge, sector extraction, feature
# engineering, outlier treatment, imputation, feature selection.
#
# USAGE:
#   python preprocessing_bangalore.py
#
#   # Or programmatically:
#   from preprocessing_bangalore import clean_bangalore_flats, clean_bangalore_houses
#   clean_bangalore_flats('bangalore_flats.csv', 'bangalore_flats_cleaned.csv')
#   clean_bangalore_houses('bangalore_houses.csv', 'bangalore_house_cleaned.csv')
# ============================================================

import re
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Reuse the description parser we already built
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper import parse_description


# ─────────────────────────────────────────────────────────────
# PRICE PARSER  →  float in Crore
# ─────────────────────────────────────────────────────────────
# Handles every format we found in the Bangalore data:
#   "1.7 Crore"            → 1.70
#   "85 Lac"               → 0.85
#   "₹85 Lac"              → 0.85
#   "₹ 0.94 - 1.34 Cr"     → 1.14   (midpoint)
#   "₹ 0.59 - 1.77 Cr"     → 1.18   (midpoint)
#   "Price on Request"     → NaN
#   "" / NaN               → NaN
# ─────────────────────────────────────────────────────────────

def parse_price_to_cr(s):
    if pd.isna(s) or not str(s).strip():
        return np.nan
    s = str(s).strip()
    if 'Price on Request' in s or 'on Request' in s:
        return np.nan

    # Strip ₹ symbol and surrounding whitespace
    s = s.replace('₹', '').strip()

    # ── RANGE format: "0.94 - 1.34 Cr"  or  "59 - 99 Lac"
    m = re.search(r'([\d.]+)\s*[-–]\s*([\d.]+)\s*(Cr|Crore|Lac|L)\b',
                  s, re.IGNORECASE)
    if m:
        lo, hi, unit = float(m.group(1)), float(m.group(2)), m.group(3).upper()
        midpoint = (lo + hi) / 2
        return round(midpoint / 100 if unit.startswith('L') else midpoint, 4)

    # ── SINGLE value: "1.7 Crore", "85 Lac", "₹85 Lac"
    m = re.search(r'([\d.]+)\s*(Crore|Cr|Lac|L)\b', s, re.IGNORECASE)
    if m:
        val, unit = float(m.group(1)), m.group(2).upper()
        return round(val / 100 if unit.startswith('L') else val, 4)

    return np.nan


# ─────────────────────────────────────────────────────────────
# BEDROOM PARSER  →  int (uses upper bound for ranges)
# ─────────────────────────────────────────────────────────────
#   "3 Bedrooms"     → 3
#   "1 Bedroom"      → 1
#   "2-3 BHK"        → 3   (upper bound — typical buyer would be looking
#                            for the larger config in a project)
#   "1-3 BHK"        → 3
#   "5"              → 5
# ─────────────────────────────────────────────────────────────

def parse_bedroom(s):
    if pd.isna(s) or not str(s).strip():
        return np.nan
    s = str(s).strip()

    # Range: "2-3 BHK", "1-4 BHK"
    m = re.search(r'(\d+)\s*[-–]\s*(\d+)\s*BHK', s, re.IGNORECASE)
    if m:
        return int(m.group(2))   # upper bound

    # Single: "3 Bedrooms", "1 Bedroom", "2 BHK"
    m = re.search(r'(\d+)', s)
    if m:
        return int(m.group(1))

    return np.nan


# ─────────────────────────────────────────────────────────────
# AREA PARSER  →  float in sqft (midpoint for ranges)
# ─────────────────────────────────────────────────────────────
#   "1,164 - 1,758 sqft"                          → 1461 (midpoint)
#   "1,114 sqft"                                  → 1114
#   "Super Built-up area 1130(104.98 sq.m.)"      → 1130
#   "Carpet area: 1250 (116.13 sq.m.)"            → 1250
#   "Built Up area: 1000 (92.9 sq.m.)"            → 1000
# ─────────────────────────────────────────────────────────────

def parse_area_to_sqft(s):
    if pd.isna(s) or not str(s).strip():
        return np.nan
    s = str(s).strip()

    # ── RANGE: "1,164 - 1,758 sqft"
    m = re.search(r'([\d,]+)\s*[-–]\s*([\d,]+)\s*sqft', s, re.IGNORECASE)
    if m:
        lo = float(m.group(1).replace(',', ''))
        hi = float(m.group(2).replace(',', ''))
        return round((lo + hi) / 2)

    # ── SUPER BUILT-UP / BUILT-UP / CARPET area types
    m = re.search(r'(?:Super Built-?up area|Built\s*Up\s*area|Carpet\s*area)'
                  r'\s*:?\s*([\d,]+)', s, re.IGNORECASE)
    if m:
        return float(m.group(1).replace(',', ''))

    # ── SINGLE sqft value: "1,114 sqft"
    m = re.search(r'([\d,]+)\s*sqft', s, re.IGNORECASE)
    if m:
        return float(m.group(1).replace(',', ''))

    # ── Fallback: any number
    m = re.search(r'([\d,]+)', s)
    if m:
        v = float(m.group(1).replace(',', ''))
        return v if v > 100 else np.nan

    return np.nan


# ─────────────────────────────────────────────────────────────
# OTHER FIELD PARSERS
# ─────────────────────────────────────────────────────────────

def parse_int_field(s):
    """For bathroom, balcony — extract integer."""
    if pd.isna(s) or not str(s).strip():
        return np.nan
    m = re.search(r'(\d+)', str(s))
    return int(m.group(1)) if m else np.nan


def parse_balcony(s):
    """Balcony: '0', '1', '2', '3+'."""
    if pd.isna(s) or not str(s).strip():
        return ''
    s = str(s).strip()
    if 'No' in s:
        return '0'
    m = re.search(r'(\d+\+?)', s)
    return m.group(1) if m else s.split()[0]


def parse_floor(s):
    """Extract numeric floor number; handle Ground/Basement/Lower."""
    if pd.isna(s) or not str(s).strip():
        return np.nan
    s = str(s).strip()
    s = re.sub(r'\bGround\b', '0', s)
    s = re.sub(r'\bBasement\b', '-1', s)
    s = re.sub(r'\bLower\b', '0', s)
    m = re.search(r'(-?\d+)', s)
    return float(m.group(1)) if m else np.nan


def clean_society(s):
    """Strip star ratings and trailing whitespace, lowercase."""
    if pd.isna(s):
        return ''
    s = re.sub(r'\d+(\.\d+)?\s?★', '', str(s)).strip().lower()
    return s if s and s != 'nan' else ''


# ─────────────────────────────────────────────────────────────
# DESCRIPTION RECOVERY
# ─────────────────────────────────────────────────────────────

def recover_from_description(row: dict) -> dict:
    """
    For project-page rows where price/bedRoom/area are missing,
    parse the description to extract these fields.
    Returns updated row dict.
    """
    desc = row.get('description', '')
    if not isinstance(desc, str) or not desc.strip():
        return row

    parsed = parse_description(desc)

    # Only fill if currently missing
    if (pd.isna(row.get('bedRoom')) or not str(row.get('bedRoom', '')).strip()) and \
       'bedRoom_max' in parsed:
        row['bedRoom'] = (f"{parsed['bedRoom_min']}-{parsed['bedRoom_max']} BHK"
                          if parsed['bedRoom_min'] != parsed['bedRoom_max']
                          else f"{parsed['bedRoom_min']} Bedrooms")

    if (pd.isna(row.get('price')) or not str(row.get('price', '')).strip()) and \
       'price_max_cr' in parsed:
        row['price'] = (f"₹ {parsed['price_min_cr']} - "
                        f"{parsed['price_max_cr']} Cr")

    if (pd.isna(row.get('areaWithType')) or
        not str(row.get('areaWithType', '')).strip()) and \
       'area_max_sqft' in parsed:
        row['areaWithType'] = (f"{parsed['area_min_sqft']:,} - "
                               f"{parsed['area_max_sqft']:,} sqft")

    return row


# ─────────────────────────────────────────────────────────────
# MAIN CLEANING — FLATS
# ─────────────────────────────────────────────────────────────

def clean_bangalore_flats(
    input_path:  str = 'bangalore_flats.csv',
    output_path: str = 'bangalore_flats_cleaned.csv',
) -> pd.DataFrame:
    """
    Bangalore-flats raw CSV → cleaned CSV matching Gurgaon's flats_cleaned.csv schema.

    Output columns (same as flats_cleaned.csv):
      property_name, property_type, society, price (Cr),
      price_per_sqft (₹/sqft), area (sqft), areaWithType, bedRoom,
      bathroom, balcony, additionalRoom, address, floorNum, facing,
      agePossession, nearbyLocations, description, furnishDetails,
      features, rating
    """
    print(f"\n[Bangalore Flats] {input_path} → {output_path}")
    df = pd.read_csv(input_path, dtype=str).fillna('')
    print(f"  Loaded {len(df)} rows")

    # ── Step 1: Recover missing fields from description
    rows = df.to_dict('records')
    rows = [recover_from_description(r) for r in rows]
    df = pd.DataFrame(rows)

    # ── Step 2: Drop rows with no price AND no bedroom AND no area
    before = len(df)
    df = df[~((df['price'].astype(str).str.strip() == '') &
              (df['bedRoom'].astype(str).str.strip() == '') &
              (df['areaWithType'].astype(str).str.strip() == ''))]
    print(f"  Dropped {before - len(df)} fully-empty rows")

    # ── Step 3: Drop duplicates by property_id
    before = len(df)
    df = df.drop_duplicates(subset=['property_id'])
    print(f"  Dropped {before - len(df)} duplicate property_ids")

    # ── Step 4: Drop unused columns
    df = df.drop(columns=['link', 'property_id'], errors='ignore')

    # ── Step 5: Parse fields
    df['society']    = df['society'].apply(clean_society)
    df['price']      = df['price'].apply(parse_price_to_cr)
    df['bedRoom']    = df['bedRoom'].apply(parse_bedroom)
    df['bathroom']   = df['bathroom'].apply(parse_int_field)
    df['balcony']    = df['balcony'].apply(parse_balcony)
    df['floorNum']   = df['floorNum'].apply(parse_floor)
    df['facing']     = df['facing'].replace('', 'NA').fillna('NA')
    df['additionalRoom'] = (df['additionalRoom']
                            .replace('', 'not available')
                            .str.lower()
                            .fillna('not available'))

    # Derived: price_per_sqft from areaWithType + price
    area_sqft = df['areaWithType'].apply(parse_area_to_sqft)
    df['price_per_sqft'] = round(
        (df['price'] * 10_000_000) / area_sqft).fillna(np.nan)
    df['area'] = area_sqft

    # ── Step 6: Drop rows with no price OR no bedroom OR no area
    before = len(df)
    df = df.dropna(subset=['price', 'bedRoom', 'area'])
    print(f"  Dropped {before - len(df)} rows missing core fields "
          f"(price/bedRoom/area)")
    df['bedRoom'] = df['bedRoom'].astype(int)

    # bathroom is allowed to be missing — we'll impute later in main pipeline
    # but for now, fill NaN with median to get integer dtype
    if df['bathroom'].notna().sum() > 0:
        median_bath = int(df['bathroom'].median())
        df['bathroom'] = df['bathroom'].fillna(median_bath).astype(int)
    else:
        df['bathroom'] = 0

    # ── Step 7: Insert property_type
    if 'property_type' not in df.columns:
        df.insert(1, 'property_type', 'flat')

    # ── Step 8: Reorder to match flats_cleaned.csv
    target_cols = [
        'property_name', 'property_type', 'society', 'price', 'price_per_sqft',
        'area', 'areaWithType', 'bedRoom', 'bathroom', 'balcony',
        'additionalRoom', 'address', 'floorNum', 'facing', 'agePossession',
        'nearbyLocations', 'description', 'furnishDetails', 'features', 'rating',
    ]
    for c in target_cols:
        if c not in df.columns:
            df[c] = ''
    df = df[target_cols]

    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved {len(df)} rows → {output_path}")

    print(f"\n  Data completeness:")
    for col in target_cols:
        if df[col].dtype == object:
            non_empty = (df[col].astype(str).str.strip() != '').sum()
        else:
            non_empty = df[col].notna().sum()
        pct = non_empty / len(df) * 100
        print(f"    {col:18s} {pct:.0f}%")

    return df


# ─────────────────────────────────────────────────────────────
# MAIN CLEANING — HOUSES
# ─────────────────────────────────────────────────────────────

def clean_bangalore_houses(
    input_path:  str = 'bangalore_houses.csv',
    output_path: str = 'bangalore_house_cleaned.csv',
) -> pd.DataFrame:
    """
    Bangalore houses raw CSV → cleaned CSV matching house_cleaned.csv schema.

    The houses CSV has 'rate' (price/sqft) and 'noOfFloor' columns
    instead of 'area' and 'floorNum' — handled here.
    """
    print(f"\n[Bangalore Houses] {input_path} → {output_path}")
    df = pd.read_csv(input_path, dtype=str).fillna('')
    print(f"  Loaded {len(df)} rows")

    # Same description recovery as flats
    rows = df.to_dict('records')
    rows = [recover_from_description(r) for r in rows]
    df = pd.DataFrame(rows)

    # Drop empty rows + dedup
    before = len(df)
    df = df[~((df['price'].astype(str).str.strip() == '') &
              (df['bedRoom'].astype(str).str.strip() == '') &
              (df['areaWithType'].astype(str).str.strip() == ''))]
    print(f"  Dropped {before - len(df)} empty rows")

    if 'property_id' in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=['property_id'])
        print(f"  Dropped {before - len(df)} duplicates")

    df = df.drop(columns=['link', 'property_id'], errors='ignore')

    # Same field parsers
    df['society']    = df['society'].apply(clean_society).replace(
                        '', 'independent').fillna('independent')
    df['price']      = df['price'].apply(parse_price_to_cr)
    df['bedRoom']    = df['bedRoom'].apply(parse_bedroom)
    df['bathroom']   = df['bathroom'].apply(parse_int_field)
    df['balcony']    = df['balcony'].apply(parse_balcony)
    df['facing']     = df['facing'].replace('', 'NA').fillna('NA')
    df['additionalRoom'] = (df['additionalRoom']
                            .replace('', 'not available')
                            .str.lower())

    # Houses use 'rate' for price-per-sqft and 'noOfFloor' for floorNum
    if 'rate' in df.columns:
        # Parse rate (₹/sqft string) to float
        df['price_per_sqft'] = df['rate'].apply(
            lambda x: float(re.sub(r'[^\d.]', '', str(x)))
            if str(x).strip() and re.search(r'\d', str(x)) else np.nan
        )
    else:
        df['price_per_sqft'] = np.nan

    df['area'] = df['areaWithType'].apply(parse_area_to_sqft)
    # Fallback: if area still missing but we have price+price_per_sqft
    fallback = df['area'].isna() & df['price'].notna() & df['price_per_sqft'].notna()
    df.loc[fallback, 'area'] = round(
        (df.loc[fallback, 'price'] * 10_000_000) / df.loc[fallback, 'price_per_sqft'])

    if 'noOfFloor' in df.columns:
        df['floorNum'] = df['noOfFloor'].apply(parse_floor)
    else:
        df['floorNum'] = np.nan

    # Drop rows missing core
    before = len(df)
    df = df.dropna(subset=['price', 'bedRoom', 'area'])
    print(f"  Dropped {before - len(df)} rows missing core fields")
    df['bedRoom'] = df['bedRoom'].astype(int)

    if df['bathroom'].notna().sum() > 0:
        df['bathroom'] = df['bathroom'].fillna(
            int(df['bathroom'].median())).astype(int)
    else:
        df['bathroom'] = 0

    if 'property_type' not in df.columns:
        df.insert(1, 'property_type', 'house')

    target_cols = [
        'property_name', 'property_type', 'society', 'price', 'price_per_sqft',
        'area', 'areaWithType', 'bedRoom', 'bathroom', 'balcony',
        'additionalRoom', 'address', 'floorNum', 'facing', 'agePossession',
        'nearbyLocations', 'description', 'furnishDetails', 'features', 'rating',
    ]
    for c in target_cols:
        if c not in df.columns:
            df[c] = ''
    df = df[target_cols]

    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved {len(df)} rows → {output_path}")
    return df


# ─────────────────────────────────────────────────────────────
# MERGE CLEANED BANGALORE WITH GURGAON
# ─────────────────────────────────────────────────────────────

def merge_with_gurgaon(
    bangalore_flats:  str = 'bangalore_flats_cleaned.csv',
    bangalore_houses: str = 'bangalore_house_cleaned.csv',
    gurgaon_flats:    str = 'flats_cleaned.csv',
    gurgaon_houses:   str = 'house_cleaned.csv',
    output_path:      str = 'all_properties.csv',
) -> pd.DataFrame:
    """
    Combine all four cleaned files into one CSV with a 'city' column.
    Output is ready for the rest of the preprocessing pipeline (sector
    extraction, feature engineering, outlier treatment, etc.).
    """
    print(f"\n[Merge] Combining all cleaned files...")
    parts = []

    for path, city in [(gurgaon_flats,    'gurgaon'),
                       (gurgaon_houses,   'gurgaon'),
                       (bangalore_flats,  'bangalore'),
                       (bangalore_houses, 'bangalore')]:
        if os.path.isfile(path):
            df = pd.read_csv(path)
            df['city'] = city
            parts.append(df)
            print(f"  + {path:35s} ({len(df)} rows, city={city})")
        else:
            print(f"  - {path:35s} (not found, skipping)")

    if not parts:
        raise FileNotFoundError("No input files found.")

    df = pd.concat(parts, ignore_index=True)
    df = df.sample(frac=1, ignore_index=True)   # shuffle
    df.to_csv(output_path, index=False)
    print(f"  ✓ Combined: {df.shape} → {output_path}")
    print(f"\n  Breakdown by city × type:")
    print(df.groupby(['city', 'property_type']).size().to_string())

    return df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Step 1: Clean Bangalore flats
    if os.path.isfile('bangalore_flats.csv'):
        clean_bangalore_flats('bangalore_flats.csv',
                              'bangalore_flats_cleaned.csv')
    else:
        print("⚠  bangalore_flats.csv not found, skipping flats")

    # Step 2: Clean Bangalore houses
    if os.path.isfile('bangalore_houses.csv'):
        clean_bangalore_houses('bangalore_houses.csv',
                               'bangalore_house_cleaned.csv')
    else:
        print("⚠  bangalore_houses.csv not found, skipping houses")

    # Step 3: Merge with Gurgaon
    print()
    if os.path.isfile('bangalore_flats_cleaned.csv') or \
       os.path.isfile('bangalore_house_cleaned.csv'):
        merge_with_gurgaon()

    print("\n✓ Bangalore preprocessing complete.")
    print("  Next: run preprocessing.py stages 3-7 on all_properties.csv")
