# ============================================================
# preprocessing.py  —  Gurgaon Properties Preprocessing Pipeline
# ============================================================
# Stages:
#   1A → flats.csv            → flats_cleaned.csv
#   1B → houses.csv           → house_cleaned.csv
#   2  → merge                → gurgaon_properties.csv
#   3  → sector + drop        → gurgaon_properties_cleaned_v1.csv
#   4  → feature engineering  → gurgaon_properties_cleaned_v2.csv
#   5  → outlier treatment    → gurgaon_properties_outlier_treated.csv
#   6  → imputation           → gurgaon_properties_missing_value_imputation.csv
#   7  → feature selection    → gurgaon_properties_post_feature_selection_v2.csv
#
# INPUT:  flats.csv, houses.csv
# INSTALL: pip install pandas numpy scikit-learn
# ============================================================

import re
import ast
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _parse_price(parts):
    """['45', 'Lac'] → 0.45 Cr  |  ['5', 'Cr'] → 5.0 Cr"""
    if isinstance(parts, float):
        return parts
    try:
        val = float(parts[0])
        return round(val / 100, 2) if parts[1] == 'Lac' else round(val, 2)
    except Exception:
        return None


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

MANUAL_SECTOR_FIXES = {
    955: 'sector 37', 2800: 'sector 92', 2838: 'sector 90', 2857: 'sector 76',
    311: 'sector 110', 1072: 'sector 110', 1486: 'sector 110',
    3040: 'sector 110', 3875: 'sector 110',
}


# ─────────────────────────────────────────────────────────────
# STAGE 1A — Clean Flats
# ─────────────────────────────────────────────────────────────

def clean_flats(input_path='flats.csv', output_path='flats_cleaned.csv'):
    """
    Raw flats.csv → flats_cleaned.csv
    Key transforms:
      - Drop link, property_id
      - Rename area → price_per_sqft
      - price: 'Price on Request' removed, Lac/Cr → float
      - price_per_sqft: strip ₹, commas → float
      - bedRoom/bathroom: extract integer
      - balcony: strip unit text
      - additionalRoom: lowercase, fill NaN
      - floorNum: extract number
      - Insert: area (derived sqft), property_type='flat'
    Output shape: ~2997 × 20
    """
    print("\n[1A] Cleaning flats...")
    df = pd.read_csv(input_path)

    df.drop(columns=['link', 'property_id'], inplace=True, errors='ignore')
    df.rename(columns={'area': 'price_per_sqft'}, inplace=True)

    df['society'] = df['society'].apply(
        lambda x: re.sub(r'\d+(\.\d+)?\s?★', '', str(x)).strip()
    ).str.lower()

    df = df[df['price'] != 'Price on Request'].copy()
    df['price'] = df['price'].str.split(' ').apply(_parse_price)

    df['price_per_sqft'] = (
        df['price_per_sqft']
        .str.split('/').str.get(0)
        .str.replace('₹', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
        .astype('float')
    )

    df = df[~df['bedRoom'].isnull()].copy()
    df['bedRoom']  = df['bedRoom'].str.split(' ').str.get(0).astype('int')
    df['bathroom'] = df['bathroom'].str.split(' ').str.get(0).astype('int')
    df['balcony']  = df['balcony'].str.split(' ').str.get(0).str.replace('No', '0')

    df['additionalRoom'] = df['additionalRoom'].fillna('not available').str.lower()

    df['floorNum'] = (
        df['floorNum']
        .astype(str)
        .str.split(' ').str.get(0)
        .str.replace('Ground', '0', regex=False)
        .str.replace('Basement', '-1', regex=False)
        .str.replace('Lower', '0', regex=False)
        .str.extract(r'(\d+)')
        .astype('float')
    )

    df['facing'] = df['facing'].fillna('NA')

    df.insert(4, 'area', round((df['price'] * 10_000_000) / df['price_per_sqft']))
    df.insert(1, 'property_type', 'flat')

    df.to_csv(output_path, index=False)
    print(f"  → {output_path}  {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# STAGE 1B — Clean Houses
# ─────────────────────────────────────────────────────────────

def clean_houses(input_path='houses.csv', output_path='house_cleaned.csv'):
    """
    Raw houses.csv → house_cleaned.csv
    Differences from flats:
      - 'rate' column instead of 'area' for price/sqft
      - 'noOfFloor' instead of 'floorNum'
      - society NaN → 'independent'
    Output shape: ~964 × 20
    """
    print("\n[1B] Cleaning houses...")
    df = pd.read_csv(input_path).drop_duplicates()

    df.drop(columns=['link', 'property_id'], inplace=True, errors='ignore')
    df.rename(columns={'rate': 'price_per_sqft'}, inplace=True)

    df['society'] = df['society'].apply(
        lambda x: re.sub(r'\d+(\.\d+)?\s?★', '', str(x)).strip()
    ).str.lower().str.replace('nan', 'independent', regex=False)

    df = df[df['price'] != 'Price on Request'].copy()
    df['price'] = df['price'].str.split(' ').apply(_parse_price)

    df['price_per_sqft'] = (
        df['price_per_sqft']
        .str.split('/').str.get(0)
        .str.replace('₹', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
        .astype('float')
    )

    df = df[~df['bedRoom'].isnull()].copy()
    df['bedRoom']  = df['bedRoom'].str.split(' ').str.get(0).astype('int')
    df['bathroom'] = df['bathroom'].str.split(' ').str.get(0).astype('int')
    df['balcony']  = df['balcony'].str.split(' ').str.get(0).str.replace('No', '0')

    df['additionalRoom'] = df['additionalRoom'].fillna('not available').str.lower()

    df.rename(columns={'noOfFloor': 'floorNum'}, inplace=True, errors='ignore')
    df['floorNum'] = df['floorNum'].astype(str).str.split(' ').str.get(0)

    df['facing'] = df['facing'].fillna('NA')

    df['area'] = round((df['price'] * 10_000_000) / df['price_per_sqft'])
    df.insert(1, 'property_type', 'house')

    df.to_csv(output_path, index=False)
    print(f"  → {output_path}  {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# STAGE 2 — Merge
# ─────────────────────────────────────────────────────────────

def merge(
    flats_path='flats_cleaned.csv',
    houses_path='house_cleaned.csv',
    output_path='gurgaon_properties.csv',
):
    """Concat + shuffle → gurgaon_properties.csv  (~3961 × 20)"""
    print("\n[2] Merging...")
    df = pd.concat(
        [pd.read_csv(flats_path), pd.read_csv(houses_path)],
        ignore_index=True
    ).sample(frac=1, ignore_index=True)
    df.to_csv(output_path, index=False)
    print(f"  → {output_path}  {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# STAGE 3 — Level-2 Preprocessing
# ─────────────────────────────────────────────────────────────

def preprocess_level2(
    input_path='gurgaon_properties.csv',
    output_path='gurgaon_properties_cleaned_v1.csv',
):
    """
    - Extracts sector from property_name (text after 'in', strip 'Gurgaon')
    - Normalises via SECTOR_MAP + manual fixes
    - Drops sectors < 3 listings
    - Drops: property_name, address, description, rating
    Output: ~3803 × 17
    """
    print("\n[3] Level-2 preprocessing (sector)...")
    df = pd.read_csv(input_path)

    df.insert(3, 'sector',
        df['property_name']
        .str.split('in').str.get(1)
        .str.replace('Gurgaon', '', regex=False)
        .str.strip()
        .str.lower()
    )

    for old, new in SECTOR_MAP.items():
        df['sector'] = df['sector'].str.replace(old, new, regex=False)

    for idx, val in MANUAL_SECTOR_FIXES.items():
        if idx in df.index:
            df.loc[idx, 'sector'] = val

    # Additional fixes found in EDA
    df['sector'] = df['sector'].str.replace('sector 95a', 'sector 95', regex=False)
    df['sector'] = df['sector'].str.replace('sector 23a', 'sector 23', regex=False)
    df['sector'] = df['sector'].str.replace('new sector 2', 'sector 110', regex=False)

    counts = df['sector'].value_counts()
    df = df[df['sector'].isin(counts[counts >= 3].index)].copy()

    df.drop(columns=['property_name', 'address', 'description', 'rating'],
            inplace=True, errors='ignore')

    df.to_csv(output_path, index=False)
    print(f"  → {output_path}  {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# STAGE 4 — Feature Engineering
# ─────────────────────────────────────────────────────────────

def _get_super_built_up(text):
    m = re.search(r'Super Built up area (\d+\.?\d*)', str(text))
    return float(m.group(1)) if m else None

def _get_area_type(text, area_type):
    m = re.search(area_type + r'\s*:\s*(\d+\.?\d*)', str(text))
    return float(m.group(1)) if m else None

def _convert_sqm_to_sqft(text, value):
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
    if pd.isna(row.get('area')) or pd.isna(row['built_up_area']):
        return row['built_up_area']
    ratio = round(row['area'] / row['built_up_area'])
    if ratio == 9.0:  return row['built_up_area'] * 9
    if ratio == 11.0: return row['built_up_area'] * 10.7
    return row['built_up_area']

def _categorize_age(value):
    if pd.isna(value): return 'Undefined'
    v = str(value)
    if any(x in v for x in ['0 to 1 Year Old', 'Within 6 months', 'Within 3 months']):
        return 'New Property'
    if '1 to 5 Year Old' in v: return 'Relatively New'
    if '5 to 10 Year Old' in v: return 'Moderately Old'
    if '10+ Year Old' in v: return 'Old Property'
    if 'Under Construction' in v or 'By' in v: return 'Under Construction'
    try:
        int(v.split(' ')[-1]); return 'Under Construction'
    except Exception: return 'Undefined'

def _get_furnishing_count(details, furnishing):
    if isinstance(details, str):
        if f'No {furnishing}' in details: return 0
        m = re.compile(f'(\\d+) {furnishing}').search(details)
        if m: return int(m.group(1))
        if furnishing in details: return 1
    return 0


def feature_engineering(
    input_path='gurgaon_properties_cleaned_v1.csv',
    output_path='gurgaon_properties_cleaned_v2.csv',
):
    """
    From gurgaon_properties_cleaned_v1.csv (17 cols) → cleaned_v2.csv (23 cols)

    New columns added:
      super_built_up_area, built_up_area, carpet_area  ← parsed from areaWithType
      study room, servant room, store room, pooja room, others  ← from additionalRoom
      furnishing_type  ← KMeans(3) on furnish detail counts
      luxury_score     ← weighted sum of binary feature flags

    Dropped: nearbyLocations, furnishDetails, features, features_list, additionalRoom
    """
    print("\n[4] Feature engineering...")
    df = pd.read_csv(input_path)

    # ── Area types from areaWithType string
    df['super_built_up_area'] = df['areaWithType'].apply(_get_super_built_up)
    df['super_built_up_area'] = df.apply(
        lambda r: _convert_sqm_to_sqft(r['areaWithType'], r['super_built_up_area']), axis=1)

    df['built_up_area'] = df['areaWithType'].apply(
        lambda x: _get_area_type(x, 'Built Up area'))
    df['built_up_area'] = df.apply(
        lambda r: _convert_sqm_to_sqft(r['areaWithType'], r['built_up_area']), axis=1)

    df['carpet_area'] = df['areaWithType'].apply(
        lambda x: _get_area_type(x, 'Carpet area'))
    df['carpet_area'] = df.apply(
        lambda r: _convert_sqm_to_sqft(r['areaWithType'], r['carpet_area']), axis=1)

    # Fix rows where all three area columns are null (plot listings)
    all_null_idx = df[
        df['super_built_up_area'].isnull() &
        df['built_up_area'].isnull() &
        df['carpet_area'].isnull()
    ].index
    tmp = df.loc[all_null_idx].copy()
    tmp['built_up_area'] = tmp['areaWithType'].apply(_extract_plot_area)
    tmp['built_up_area'] = tmp.apply(_fix_scale, axis=1)
    df.update(tmp)

    # ── Additional rooms → binary flags
    for col in ['study room', 'servant room', 'store room', 'pooja room', 'others']:
        df[col] = df['additionalRoom'].str.contains(col, na=False).astype(int)

    # ── Age of possession
    df['agePossession'] = df['agePossession'].apply(_categorize_age)

    # ── Furnishing type via KMeans on furnish detail counts
    all_furnishings = []
    for detail in df['furnishDetails'].dropna():
        parts = str(detail).replace('[','').replace(']','').replace("'",'').split(', ')
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
    df['furnishing_type'] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(scaled)
    # 0 = unfurnished, 1 = semifurnished, 2 = furnished

    # ── Luxury score from features column
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

    # ── Cleanup
    df.drop(columns=['nearbyLocations', 'furnishDetails', 'features',
                     'features_list', 'additionalRoom'],
            inplace=True, errors='ignore')

    df.to_csv(output_path, index=False)
    print(f"  → {output_path}  {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# STAGE 5 — Outlier Treatment
# ─────────────────────────────────────────────────────────────

def treat_outliers(
    input_path='gurgaon_properties_cleaned_v2.csv',
    output_path='gurgaon_properties_outlier_treated.csv',
):
    """
    From cleaned_v2 (3803 × 23) → outlier_treated (3555 × 24)

    Steps:
      1. Fix price_per_sqft scale errors (×9 if area<1000); cap at 50k
      2. Drop area > 100k and 9 known bad rows
      3. Manual area corrections for 8 rows
      4. Cap bedRoom ≤ 10
      5. Fix carpet_area for row 2131
      6. Recalculate price_per_sqft after fixes
      7. Add area_room_ratio
    """
    print("\n[5] Outlier treatment...")
    df = pd.read_csv(input_path).drop_duplicates()

    # price_per_sqft: fix scale errors then cap
    Q1, Q3 = df['price_per_sqft'].quantile([0.25, 0.75])
    outliers_sqft = df[df['price_per_sqft'] > Q3 + 1.5 * (Q3 - Q1)].copy()
    outliers_sqft['area'] = outliers_sqft['area'].apply(lambda x: x * 9 if x < 1000 else x)
    outliers_sqft['price_per_sqft'] = round(
        (outliers_sqft['price'] * 10_000_000) / outliers_sqft['area'])
    df.update(outliers_sqft)
    df = df[df['price_per_sqft'] <= 50000].copy()

    # area: cap + drop bad rows
    df = df[df['area'] < 100_000].copy()
    bad = [818, 1796, 1123, 2, 2356, 115, 3649, 2503, 1471]
    df.drop(index=[i for i in bad if i in df.index], inplace=True)

    # Manual area corrections (verified during EDA)
    fixes = {48: 115*9, 300: 7250, 2666: 5800, 1358: 2660,
             3195: 2850, 2131: 1812, 3088: 2160, 3444: 1175}
    for idx, val in fixes.items():
        if idx in df.index:
            df.loc[idx, 'area'] = val

    df = df[df['bedRoom'] <= 10].copy()

    if 2131 in df.index:
        df.loc[2131, 'carpet_area'] = 1812

    df['price_per_sqft'] = round((df['price'] * 10_000_000) / df['area'])
    df['area_room_ratio'] = df['area'] / df['bedRoom']

    df.to_csv(output_path, index=False)
    print(f"  → {output_path}  {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# STAGE 6 — Missing Value Imputation
# ─────────────────────────────────────────────────────────────

def _mode_impute_age(df_ref, row, groupby_cols):
    if row['agePossession'] != 'Undefined':
        return row['agePossession']
    mask = pd.Series([True] * len(df_ref), index=df_ref.index)
    for col in groupby_cols:
        mask &= (df_ref[col] == row[col])
    mode = df_ref.loc[mask, 'agePossession'].mode()
    return mode.iloc[0] if not mode.empty else row['agePossession']


def impute_missing(
    input_path='gurgaon_properties_outlier_treated.csv',
    output_path='gurgaon_properties_missing_value_imputation.csv',
):
    """
    From outlier_treated (3555 × 24) → missing_value_imputation (3554 × 18)

    Steps:
      1. Compute super_built_up:built_up and carpet:built_up median ratios
      2. Impute built_up_area using available area columns + ratios
      3. Fix built_up_area anomaly (small area, high price)
      4. Drop area, areaWithType, super_built_up_area, carpet_area, area_room_ratio
      5. floorNum: fillna(2.0)
      6. Drop facing column
      7. Drop any remaining null rows
      8. Three-pass mode imputation for agePossession='Undefined'
    """
    print("\n[6] Missing value imputation...")
    df = pd.read_csv(input_path)

    complete = df[
        ~df['super_built_up_area'].isnull() &
        ~df['built_up_area'].isnull() &
        ~df['carpet_area'].isnull()
    ]
    r_super  = (complete['super_built_up_area'] / complete['built_up_area']).median()
    r_carpet = (complete['carpet_area']         / complete['built_up_area']).median()

    # Case 1: super + carpet present
    m = (~df['super_built_up_area'].isnull() &
          df['built_up_area'].isnull() &
         ~df['carpet_area'].isnull())
    tmp = df[m].copy()
    tmp['built_up_area'] = round(
        (tmp['super_built_up_area'] / r_super + tmp['carpet_area'] / r_carpet) / 2)
    df.update(tmp)

    # Case 2: only super present
    m = (~df['super_built_up_area'].isnull() &
          df['built_up_area'].isnull() &
          df['carpet_area'].isnull())
    tmp = df[m].copy()
    tmp['built_up_area'] = round(tmp['super_built_up_area'] / r_super)
    df.update(tmp)

    # Case 3: only carpet present
    m = (df['super_built_up_area'].isnull() &
         df['built_up_area'].isnull() &
        ~df['carpet_area'].isnull())
    tmp = df[m].copy()
    tmp['built_up_area'] = round(tmp['carpet_area'] / r_carpet)
    df.update(tmp)

    # Fix anomaly: small built_up_area paired with high price → use raw area
    anom = df[(df['built_up_area'] < 2000) & (df['price'] > 2.5)].copy()
    anom['built_up_area'] = anom['area']
    df.update(anom)

    df.drop(columns=['area', 'areaWithType', 'super_built_up_area',
                     'carpet_area', 'area_room_ratio'],
            inplace=True, errors='ignore')

    df['floorNum'] = df['floorNum'].fillna(2.0)
    df.drop(columns=['facing'], inplace=True, errors='ignore')
    df.dropna(inplace=True)

    # 3-pass mode imputation for agePossession
    for groupby in [['sector', 'property_type'], ['sector'], ['property_type']]:
        df['agePossession'] = df.apply(
            lambda r: _mode_impute_age(df, r, groupby), axis=1)

    df.to_csv(output_path, index=False)
    print(f"  → {output_path}  {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# STAGE 7 — Feature Selection
# ─────────────────────────────────────────────────────────────

def _categorize_luxury(score):
    if 0 <= score < 50:   return 'Low'
    if 50 <= score < 150: return 'Medium'
    if score <= 175:      return 'High'
    return 'Low'

def _categorize_floor(floor):
    try:
        f = float(floor)
        if 0 <= f <= 2:   return 'Low Floor'
        if 3 <= f <= 10:  return 'Mid Floor'
        if 11 <= f <= 51: return 'High Floor'
    except Exception:
        pass
    return 'Low Floor'


def feature_selection(
    input_path='gurgaon_properties_missing_value_imputation.csv',
    output_path='gurgaon_properties_post_feature_selection_v2.csv',
):
    """
    From missing_value_imputation (3554 × 18) → post_feature_selection_v2 (3554 × 13)
    (or × 14 if 'city' column is present, e.g. for the combined Gurgaon+Bangalore dataset)

    Added:
      luxury_category  ← Low/Medium/High from luxury_score
      floor_category   ← Low/Mid/High Floor from floorNum

    Dropped (confirmed low importance via RF, SHAP, permutation, LASSO):
      pooja room, study room, others, floorNum, luxury_score, society, price_per_sqft

    Final columns (string categoricals = _v2 format):
      city (if present), property_type, sector, price, bedRoom, bathroom,
      balcony, agePossession, built_up_area, servant room, store room,
      furnishing_type, luxury_category, floor_category
    """
    print("\n[7] Feature selection...")
    df = pd.read_csv(input_path)

    df['luxury_category'] = df['luxury_score'].apply(_categorize_luxury)
    df['floor_category']  = df['floorNum'].apply(_categorize_floor)

    df.drop(columns=['pooja room', 'study room', 'others',
                     'floorNum', 'luxury_score', 'society', 'price_per_sqft'],
            inplace=True, errors='ignore')

    # Reorder columns to match _v2 format (used by model.py)
    # Include 'city' as the leading column if present (multi-city dataset)
    base_cols = ['property_type', 'sector', 'price', 'bedRoom', 'bathroom',
                 'balcony', 'agePossession', 'built_up_area', 'servant room',
                 'store room', 'furnishing_type', 'luxury_category',
                 'floor_category']

    if 'city' in df.columns:
        col_order = ['city'] + base_cols
    else:
        col_order = base_cols

    df = df[[c for c in col_order if c in df.columns]]

    df.to_csv(output_path, index=False)
    print(f"  → {output_path}  {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# RUN FULL PIPELINE
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    clean_flats()
    clean_houses()
    merge()
    preprocess_level2()
    feature_engineering()
    treat_outliers()
    impute_missing()
    feature_selection()
    print("\n✓ All stages complete.")
    print("  Final file: gurgaon_properties_post_feature_selection_v2.csv")
