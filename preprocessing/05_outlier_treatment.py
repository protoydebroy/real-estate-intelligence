"""
Stage 5 — Outlier Treatment
============================
Caps & corrects extreme values that would otherwise dominate the model.

INPUT:   data/processed/all_v2.csv (5,377 × 24)
OUTPUT:  data/processed/all_outlier.csv (~5,126 × 25)

WHAT IT DOES:
  1. price_per_sqft outliers above Q3+1.5×IQR:
     - Often a scale error (sqyd vs sqft); fix area ×9 if area<1000
     - Cap at ₹50,000/sqft after fix

  2. Drop area > 100,000 sqft (impossible for residential)
  3. Drop 9 known bad rows (verified during EDA)
  4. Manual area corrections for 8 rows
  5. Cap bedRoom ≤ 10
  6. Fix carpet_area for one specific row
  7. Recalculate price_per_sqft after all area fixes
  8. Add area_room_ratio as a derived feature

USAGE:
    python preprocessing/05_outlier_treatment.py
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd


def treat_outliers(
    input_path:  str = 'data/processed/all_v2.csv',
    output_path: str = 'data/processed/all_outlier.csv',
) -> pd.DataFrame:

    print(f"\n[Stage 5] Outlier treatment → {output_path}")
    df = pd.read_csv(input_path).drop_duplicates()
    n0 = len(df)
    print(f"  Loaded: {df.shape}")

    # ── 1. Fix price_per_sqft scale errors then cap
    Q1, Q3 = df['price_per_sqft'].quantile([0.25, 0.75])
    iqr_thresh = Q3 + 1.5 * (Q3 - Q1)
    sqft_outliers = df[df['price_per_sqft'] > iqr_thresh].copy()
    sqft_outliers['area'] = sqft_outliers['area'].apply(
        lambda x: x * 9 if x < 1000 else x)
    sqft_outliers['price_per_sqft'] = round(
        (sqft_outliers['price'] * 10_000_000) / sqft_outliers['area'])
    df.update(sqft_outliers)
    df = df[df['price_per_sqft'] <= 50_000].copy()

    # ── 2. Cap area + drop known bad rows
    df = df[df['area'] < 100_000].copy()
    bad_rows = [818, 1796, 1123, 2, 2356, 115, 3649, 2503, 1471]
    df.drop(index=[i for i in bad_rows if i in df.index], inplace=True)

    # ── 3. Manual area corrections (verified during EDA)
    fixes = {48: 115*9, 300: 7250, 2666: 5800, 1358: 2660,
             3195: 2850, 2131: 1812, 3088: 2160, 3444: 1175}
    for idx, val in fixes.items():
        if idx in df.index:
            df.loc[idx, 'area'] = val

    # ── 4. Cap bedrooms
    df = df[df['bedRoom'] <= 10].copy()

    # ── 5. Specific row fix
    if 2131 in df.index:
        df.loc[2131, 'carpet_area'] = 1812

    # ── 6. Recalculate price_per_sqft after corrections
    df['price_per_sqft'] = round((df['price'] * 10_000_000) / df['area'])

    # ── 7. Derived feature
    df['area_room_ratio'] = df['area'] / df['bedRoom']

    df.to_csv(output_path, index=False)
    print(f"  Dropped {n0 - len(df)} outlier rows")
    print(f"  ✓ Saved → {output_path}  shape={df.shape}")
    return df


if __name__ == '__main__':
    treat_outliers()
    print("\n✓ Stage 5 complete. Next: python preprocessing/06_impute_missing.py")
