"""
Stage 4 — Feature Engineering
==============================
Creates numerical & categorical features from the messy text columns.

INPUT:   data/processed/all_v1.csv (5,377 × 18)
OUTPUT:  data/processed/all_v2.csv (5,377 × 24)

WHAT IT DOES:
  1. Parses 'areaWithType' string into:
       super_built_up_area, built_up_area, carpet_area
     (handles sqm → sqft conversion, plot area for houses)

  2. Extracts 5 binary flags from 'additionalRoom':
       study room, servant room, store room, pooja room, others

  3. Buckets 'agePossession' into 5 categories:
       New Property, Relatively New, Moderately Old,
       Old Property, Under Construction

  4. KMeans-clusters furnish details into 'furnishing_type':
       0 = unfurnished, 1 = semifurnished, 2 = furnished

  5. Computes 'luxury_score' as weighted sum of binary feature flags
     (Golf Course = 10, Spa = 9, Gym = 8, ... ATM = 4)

DROPS: nearbyLocations, furnishDetails, features, additionalRoom

USAGE:
    python preprocessing/04_feature_engineering.py
"""

import re
import ast
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.cluster import KMeans


# ─────────────────────────────────────────────────────────────
# LUXURY FEATURE WEIGHTS
# Each amenity contributes (1 if present × weight) to luxury_score
# Higher weight = more luxurious feature
# ─────────────────────────────────────────────────────────────

LUXURY_WEIGHTS = {
    '24/7 Power Backup': 8, '24/7 Water Supply': 4, '24x7 Security': 7,
    'ATM': 4, 'Aerobics Centre': 6, 'Airy Rooms': 8, 'Amphitheatre': 7,
    'Badminton Court': 7, 'Banquet Hall': 8, 'Bar/Chill-Out Lounge': 9,
    'Barbecue': 7, 'Basketball Court': 7, 'Billiards': 7, 'Bowling Alley': 8,
    'Business Lounge': 9, 'CCTV Camera Security': 8, 'Cafeteria': 6,
    'Car Parking': 6, 'Card Room': 6, 'Centrally Air Conditioned': 9,
    'Changing Area': 6, "Children's Play Area": 7, 'Cigar Lounge': 9,
    'Clinic': 5, 'Club House': 9, 'Concierge Service': 9, 'Conference room': 8,
    'Creche/Day care': 7, 'Cricket Pitch': 7, 'Doctor on Call': 6,
    'Earthquake Resistant': 5, 'Entrance Lobby': 7, 'False Ceiling Lighting': 6,
    'Feng Shui / Vaastu Compliant': 5, 'Fire Fighting Systems': 8,
    'Fitness Centre / GYM': 8, 'Flower Garden': 7, 'Food Court': 6,
    'Foosball': 5, 'Football': 7, 'Fountain': 7, 'Gated Community': 7,
    'Golf Course': 10, 'Grocery Shop': 6, 'Gymnasium': 8,
    'High Ceiling Height': 8, 'High Speed Elevators': 8, 'Infinity Pool': 9,
    'Intercom Facility': 7, 'Internal Street Lights': 6,
    'Internet/wi-fi connectivity': 7, 'Jacuzzi': 9, 'Jogging Track': 7,
    'Landscape Garden': 8, 'Laundry': 6, 'Lawn Tennis Court': 8,
    'Library': 8, 'Lounge': 8, 'Low Density Society': 7,
    'Maintenance Staff': 6, 'Manicured Garden': 7, 'Medical Centre': 5,
    'Milk Booth': 4, 'Mini Theatre': 9, 'Multipurpose Court': 7,
    'Multipurpose Hall': 7, 'Natural Light': 8, 'Natural Pond': 7,
    'Park': 8, 'Party Lawn': 8, 'Piped Gas': 7, 'Pool Table': 7,
    'Power Back up Lift': 8, 'Private Garden / Terrace': 9,
    'Property Staff': 7, 'RO System': 7, 'Rain Water Harvesting': 7,
    'Reading Lounge': 8, 'Restaurant': 8, 'Salon': 8, 'Sauna': 9,
    'Security / Fire Alarm': 9, 'Security Personnel': 9,
    'Separate entry for servant room': 8, 'Sewage Treatment Plant': 6,
    'Shopping Centre': 7, 'Skating Rink': 7, 'Solar Lighting': 6,
    'Solar Water Heating': 7, 'Spa': 9, 'Spacious Interiors': 9,
    'Squash Court': 8, 'Steam Room': 9, 'Sun Deck': 8, 'Swimming Pool': 8,
    'Temple': 5, 'Theatre': 9, 'Toddler Pool': 7, 'Valet Parking': 9,
    'Video Door Security': 9, 'Visitor Parking': 7, 'Water Softener Plant': 7,
    'Water Storage': 7, 'Water purifier': 7, 'Yoga/Meditation Area': 7,
}


# ─────────────────────────────────────────────────────────────
# AREA PARSERS
# ─────────────────────────────────────────────────────────────

def _get_super_built_up(text):
    m = re.search(r'Super Built up area (\d+\.?\d*)', str(text))
    return float(m.group(1)) if m else None


def _get_area_type(text, area_type):
    m = re.search(area_type + r'\s*:\s*(\d+\.?\d*)', str(text))
    return float(m.group(1)) if m else None


def _convert_sqm_to_sqft(text, value):
    """If areaWithType has '... (XX sq.m.)' alongside the sqft value,
    use the sqm value × 10.7639 since it's the canonical one."""
    if value is None or pd.isna(value):
        return value
    try:
        val_str = str(int(value)) if value == int(value) else str(value)
    except (ValueError, TypeError):
        return value
    m = re.search(r'{} \((\d+\.?\d*) sq\.m\.\)'.format(val_str), str(text))
    return float(m.group(1)) * 10.7639 if m else value


def _extract_plot_area(text):
    m = re.search(r'Plot area (\d+\.?\d*)', str(text))
    return float(m.group(1)) if m else None


def _fix_scale(row):
    """Plot-area sometimes mis-scaled (gaj → sqft = ×9, sqm → sqft = ×10.7)."""
    if pd.isna(row.get('area')) or pd.isna(row['built_up_area']):
        return row['built_up_area']
    ratio = round(row['area'] / row['built_up_area'])
    if ratio == 9.0:
        return row['built_up_area'] * 9
    if ratio == 11.0:
        return row['built_up_area'] * 10.7
    return row['built_up_area']


def _categorize_age(value):
    if pd.isna(value):
        return 'Undefined'
    v = str(value)
    if any(x in v for x in ['0 to 1 Year Old', 'Within 6 months',
                             'Within 3 months']):
        return 'New Property'
    if '1 to 5 Year Old'  in v: return 'Relatively New'
    if '5 to 10 Year Old' in v: return 'Moderately Old'
    if '10+ Year Old'     in v: return 'Old Property'
    if 'Under Construction' in v or 'By' in v: return 'Under Construction'
    try:
        int(v.split(' ')[-1])
        return 'Under Construction'
    except Exception:
        return 'Undefined'


def _get_furnishing_count(details, furnishing):
    """Extract numeric count of a specific furnishing item from text."""
    if isinstance(details, str):
        if f'No {furnishing}' in details:
            return 0
        m = re.compile(f'(\\d+) {furnishing}').search(details)
        if m:
            return int(m.group(1))
        if furnishing in details:
            return 1
    return 0


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def feature_engineering(
    input_path:  str = 'data/processed/all_v1.csv',
    output_path: str = 'data/processed/all_v2.csv',
) -> pd.DataFrame:

    print(f"\n[Stage 4] Feature engineering → {output_path}")
    df = pd.read_csv(input_path)
    print(f"  Loaded: {df.shape}")

    # ── 1. Area types from areaWithType
    df['super_built_up_area'] = df['areaWithType'].apply(_get_super_built_up)
    df['super_built_up_area'] = df.apply(
        lambda r: _convert_sqm_to_sqft(r['areaWithType'],
                                       r['super_built_up_area']), axis=1)

    df['built_up_area'] = df['areaWithType'].apply(
        lambda x: _get_area_type(x, 'Built Up area'))
    df['built_up_area'] = df.apply(
        lambda r: _convert_sqm_to_sqft(r['areaWithType'],
                                       r['built_up_area']), axis=1)

    df['carpet_area'] = df['areaWithType'].apply(
        lambda x: _get_area_type(x, 'Carpet area'))
    df['carpet_area'] = df.apply(
        lambda r: _convert_sqm_to_sqft(r['areaWithType'],
                                       r['carpet_area']), axis=1)

    # Fix plot-area rows (independent houses)
    all_null_idx = df[
        df['super_built_up_area'].isnull() &
        df['built_up_area'].isnull() &
        df['carpet_area'].isnull()
    ].index
    tmp = df.loc[all_null_idx].copy()
    tmp['built_up_area'] = tmp['areaWithType'].apply(_extract_plot_area)
    tmp['built_up_area'] = tmp.apply(_fix_scale, axis=1)
    df.update(tmp)

    # ── 2. Additional rooms → 5 binary flags
    for col in ['study room', 'servant room', 'store room',
                'pooja room', 'others']:
        df[col] = df['additionalRoom'].str.contains(col, na=False).astype(int)

    # ── 3. Age category bucketing
    df['agePossession'] = df['agePossession'].apply(_categorize_age)

    # ── 4. KMeans furnishing clusters (3 levels)
    all_furnishings = []
    for detail in df['furnishDetails'].dropna():
        parts = (str(detail).replace('[', '').replace(']', '')
                 .replace("'", '').split(', '))
        all_furnishings.extend(parts)

    unique_furnishings = list({
        re.sub(r'No |\d+', '', f).strip()
        for f in set(all_furnishings)
        if re.sub(r'No |\d+', '', f).strip()
    })

    furn_matrix = pd.DataFrame(index=df.index)
    for furn in unique_furnishings:
        furn_matrix[furn] = df['furnishDetails'].apply(
            lambda x: _get_furnishing_count(x, furn))

    scaled = StandardScaler().fit_transform(furn_matrix)
    df['furnishing_type'] = KMeans(
        n_clusters=3, random_state=42, n_init=10).fit_predict(scaled)

    # ── 5. Luxury score
    df['features_list'] = df['features'].apply(
        lambda x: ast.literal_eval(x)
        if pd.notnull(x) and str(x).startswith('[') else []
    )
    mlb = MultiLabelBinarizer()
    feat_bin = pd.DataFrame(
        mlb.fit_transform(df['features_list']),
        columns=mlb.classes_, index=df.index,
    )
    common = [c for c in LUXURY_WEIGHTS if c in feat_bin.columns]
    df['luxury_score'] = feat_bin[common].multiply(
        [LUXURY_WEIGHTS[c] for c in common]).sum(axis=1)

    # Cleanup
    df.drop(columns=['nearbyLocations', 'furnishDetails', 'features',
                     'features_list', 'additionalRoom'],
            inplace=True, errors='ignore')

    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved → {output_path}  shape={df.shape}")
    return df


if __name__ == '__main__':
    feature_engineering()
    print("\n✓ Stage 4 complete. Next: python preprocessing/05_outlier_treatment.py")
