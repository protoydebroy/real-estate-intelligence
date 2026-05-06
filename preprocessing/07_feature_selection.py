"""
Stage 7 — Feature Selection
============================
Final stage: discretise continuous features into ordinal categories and
drop columns confirmed to be low-importance.

INPUT:   data/processed/all_imputed.csv (~4,543 × 19)
OUTPUT:  data/processed/all_final_v2.csv (~4,543 × 14)  ← model-ready

WHAT IT DOES:
  1. Adds 'luxury_category' from luxury_score:
       Low    : score < 50
       Medium : 50 ≤ score < 150
       High   : 150 ≤ score ≤ 175

  2. Adds 'floor_category' from floorNum:
       Low Floor  : 0–2
       Mid Floor  : 3–10
       High Floor : 11–51

  3. Drops columns confirmed low-importance via consensus across:
       - Random Forest feature importance
       - SHAP values
       - Permutation importance
       - LASSO coefficients
     Dropped: pooja room, study room, others,
              floorNum (replaced by floor_category),
              luxury_score (replaced by luxury_category),
              society (high cardinality, low signal),
              price_per_sqft (target leakage — derived from price)

  4. Reorders to the canonical 14-column model-ready format:
       city, property_type, sector, price, bedRoom, bathroom,
       balcony, agePossession, built_up_area, servant room,
       store room, furnishing_type, luxury_category, floor_category

USAGE:
    python preprocessing/07_feature_selection.py
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd


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
    input_path:  str = 'data/processed/all_imputed.csv',
    output_path: str = 'data/processed/all_final_v2.csv',
) -> pd.DataFrame:

    print(f"\n[Stage 7] Feature selection → {output_path}")
    df = pd.read_csv(input_path)
    print(f"  Loaded: {df.shape}")

    # ── 1 & 2. Discretise luxury and floor
    df['luxury_category'] = df['luxury_score'].apply(_categorize_luxury)
    df['floor_category']  = df['floorNum'].apply(_categorize_floor)

    # ── 3. Drop low-importance columns
    drop_cols = ['pooja room', 'study room', 'others',
                 'floorNum', 'luxury_score', 'society', 'price_per_sqft']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    print(f"  Dropped: {drop_cols}")

    # ── 4. Reorder to canonical model-ready format
    base_cols = ['property_type', 'sector', 'price', 'bedRoom', 'bathroom',
                 'balcony', 'agePossession', 'built_up_area', 'servant room',
                 'store room', 'furnishing_type', 'luxury_category',
                 'floor_category']
    col_order = (['city'] + base_cols) if 'city' in df.columns else base_cols
    df = df[[c for c in col_order if c in df.columns]]

    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved → {output_path}  shape={df.shape}")
    print(f"  Columns: {df.columns.tolist()}")

    if 'city' in df.columns:
        print(f"\n  Per city × property_type:")
        print(df.groupby(['city', 'property_type']).size().to_string())

    return df


if __name__ == '__main__':
    feature_selection()
    print("\n✓ Stage 7 complete. Pipeline finished!")
    print("  Next: python model/train_model.py")
