"""
Stage 6 — Missing Value Imputation
====================================
Fills missing built_up_area using ratio-based inference and mode-fills
agePossession via group-wise lookups.

INPUT:   data/processed/all_outlier.csv (~5,126 × 25)
OUTPUT:  data/processed/all_imputed.csv (~4,543 × 19)

WHAT IT DOES:
  1. Compute median ratios from rows where ALL THREE areas are present:
       r_super  = super_built_up / built_up    (median across complete rows)
       r_carpet = carpet         / built_up    (median across complete rows)

  2. For rows missing built_up_area, infer it from whichever areas ARE present:
       Case 1: super + carpet present → average of (super/r_super, carpet/r_carpet)
       Case 2: only super present     → super / r_super
       Case 3: only carpet present    → carpet / r_carpet

  3. Fix anomalies where built_up < 2000 sqft yet price > ₹2.5 Cr (suspicious
     for premium properties) — replace built_up with the raw 'area' column

  4. Drop the now-redundant area columns:
       area, areaWithType, super_built_up_area, carpet_area, area_room_ratio

  5. floorNum.fillna(2.0)  (median) and drop the noisy 'facing' column

  6. Three-pass mode imputation for 'Undefined' agePossession:
       Pass 1: fill from {sector + property_type} group mode
       Pass 2: fill remaining from {sector} group mode
       Pass 3: fill remaining from {property_type} group mode

USAGE:
    python preprocessing/06_impute_missing.py
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd


def _mode_impute_age(df_ref, row, groupby_cols):
    if row['agePossession'] != 'Undefined':
        return row['agePossession']
    mask = pd.Series([True] * len(df_ref), index=df_ref.index)
    for col in groupby_cols:
        mask &= (df_ref[col] == row[col])
    mode = df_ref.loc[mask, 'agePossession'].mode()
    return mode.iloc[0] if not mode.empty else row['agePossession']


def impute_missing(
    input_path:  str = 'data/processed/all_outlier.csv',
    output_path: str = 'data/processed/all_imputed.csv',
) -> pd.DataFrame:

    print(f"\n[Stage 6] Missing value imputation → {output_path}")
    df = pd.read_csv(input_path)
    print(f"  Loaded: {df.shape}")

    # ── 1. Compute median ratios from fully-complete rows
    complete = df[
        ~df['super_built_up_area'].isnull() &
        ~df['built_up_area'].isnull() &
        ~df['carpet_area'].isnull()
    ]
    r_super  = (complete['super_built_up_area'] / complete['built_up_area']).median()
    r_carpet = (complete['carpet_area']         / complete['built_up_area']).median()
    print(f"  Median ratios — super:built = {r_super:.3f}, "
          f"carpet:built = {r_carpet:.3f}")

    # ── 2. Three imputation cases
    # Case 1: super + carpet present, built_up missing
    m = (~df['super_built_up_area'].isnull() &
          df['built_up_area'].isnull() &
         ~df['carpet_area'].isnull())
    if m.any():
        tmp = df[m].copy()
        tmp['built_up_area'] = round(
            (tmp['super_built_up_area'] / r_super +
             tmp['carpet_area']         / r_carpet) / 2)
        df.update(tmp)
        print(f"  Case 1 (super+carpet): {m.sum()} rows imputed")

    # Case 2: only super present
    m = (~df['super_built_up_area'].isnull() &
          df['built_up_area'].isnull() &
          df['carpet_area'].isnull())
    if m.any():
        tmp = df[m].copy()
        tmp['built_up_area'] = round(tmp['super_built_up_area'] / r_super)
        df.update(tmp)
        print(f"  Case 2 (super only):   {m.sum()} rows imputed")

    # Case 3: only carpet present
    m = (df['super_built_up_area'].isnull() &
         df['built_up_area'].isnull() &
        ~df['carpet_area'].isnull())
    if m.any():
        tmp = df[m].copy()
        tmp['built_up_area'] = round(tmp['carpet_area'] / r_carpet)
        df.update(tmp)
        print(f"  Case 3 (carpet only):  {m.sum()} rows imputed")

    # ── 3. Fix premium-property anomaly
    anomaly = df[(df['built_up_area'] < 2000) & (df['price'] > 2.5)].copy()
    if len(anomaly):
        anomaly['built_up_area'] = anomaly['area']
        df.update(anomaly)
        print(f"  Fixed {len(anomaly)} premium-property anomalies "
              f"(built_up < 2000 but price > ₹2.5 Cr)")

    # ── 4. Drop redundant columns
    df.drop(columns=['area', 'areaWithType', 'super_built_up_area',
                     'carpet_area', 'area_room_ratio'],
            inplace=True, errors='ignore')

    # ── 5. floorNum + facing
    df['floorNum'] = df['floorNum'].fillna(2.0)
    df.drop(columns=['facing'], inplace=True, errors='ignore')

    # Drop any remaining null rows
    n0 = len(df)
    df.dropna(inplace=True)
    print(f"  Dropped {n0 - len(df)} rows with remaining nulls")

    # ── 6. Three-pass mode imputation for agePossession
    for groupby in [['sector', 'property_type'], ['sector'], ['property_type']]:
        df['agePossession'] = df.apply(
            lambda r: _mode_impute_age(df, r, groupby), axis=1)

    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved → {output_path}  shape={df.shape}")
    return df


if __name__ == '__main__':
    impute_missing()
    print("\n✓ Stage 6 complete. Next: python preprocessing/07_feature_selection.py")
