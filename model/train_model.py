"""
train_model.py  —  Price Prediction Model Training
===================================================
Trains the property price prediction model on the multi-city dataset.

INPUT:   data/processed/all_final_v2.csv  (4,555 × 14)
OUTPUT:  model/pipeline.pkl  (sklearn Pipeline — preprocessor + regressor)
         model/df.pkl        (feature DataFrame X — used by app for option lookup)

WHAT IT DOES:
  1. Loads the model-ready CSV
  2. Splits into X (13 features) and y (price in Cr)
  3. Log-transforms y → np.log1p(price)        (real-estate prices are skewed)
  4. Builds a ColumnTransformer:
       num : StandardScaler
       cat : OneHotEncoder(drop='first', handle_unknown='ignore')
  5. Benchmarks Random Forest vs XGBoost using 10-fold CV
  6. Picks the winner, fits on the full dataset, saves to disk

USAGE:
    python model/train_model.py            # full benchmark + train + save
    python model/train_model.py --quick    # use 100 trees for fast iteration
    python model/train_model.py --skip-cv  # skip CV, just train final
"""

import os
import sys
import pickle
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.compose         import ColumnTransformer
from sklearn.preprocessing   import StandardScaler, OneHotEncoder
from sklearn.pipeline        import Pipeline
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics         import mean_absolute_error, r2_score

import xgboost as xgb


# ─────────────────────────────────────────────────────────────
# PATHS  (relative to repo root, regardless of where script runs from)
# ─────────────────────────────────────────────────────────────

HERE       = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(HERE, '..'))

DATA_PATH      = os.path.join(REPO_ROOT, 'data', 'processed', 'all_final_v2.csv')
PIPELINE_PATH  = os.path.join(REPO_ROOT, 'model', 'pipeline.pkl')
DF_PATH        = os.path.join(REPO_ROOT, 'model', 'df.pkl')


# ─────────────────────────────────────────────────────────────
# FEATURE COLUMNS
# ─────────────────────────────────────────────────────────────

NUM_COLS = ['bedRoom', 'bathroom', 'built_up_area',
            'servant room', 'store room']

# 'city' is included if present (multi-city dataset). Falls back gracefully
# to single-city when the column is absent.
CAT_COLS_BASE = ['property_type', 'sector', 'balcony', 'agePossession',
                 'furnishing_type', 'luxury_category', 'floor_category']

KFOLD = KFold(n_splits=10, shuffle=True, random_state=42)


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_data(path: str = DATA_PATH):
    """
    Load the model-ready CSV.

    furnishing_type may be 0/1/2 (numeric, from the pipeline) — convert
    to readable strings so OHE produces interpretable feature names.

    Returns: (X, y, full_df)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Run preprocessing first:\n"
            f"  python preprocessing/run_all.py"
        )

    df = pd.read_csv(path)

    df['furnishing_type'] = df['furnishing_type'].replace(
        {0.0: 'unfurnished', 1.0: 'semifurnished', 2.0: 'furnished',
         0:   'unfurnished', 1:   'semifurnished', 2:   'furnished'}
    )

    X = df.drop(columns=['price'])
    y = df['price']
    return X, y, df


def _get_cat_cols(X) -> list:
    """Categorical columns present in X (includes 'city' if available)."""
    cols = ['city'] + CAT_COLS_BASE if 'city' in X.columns else CAT_COLS_BASE
    return [c for c in cols if c in X.columns]


# ─────────────────────────────────────────────────────────────
# PREPROCESSOR
# ─────────────────────────────────────────────────────────────

def build_preprocessor(cat_cols=None):
    """
    ColumnTransformer:
      NUM_COLS → StandardScaler
      cat_cols → OneHotEncoder(drop='first', handle_unknown='ignore')
    """
    if cat_cols is None:
        cat_cols = CAT_COLS_BASE
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUM_COLS),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore',
                                  sparse_output=False), cat_cols),
        ],
        remainder='passthrough',
    )


# ─────────────────────────────────────────────────────────────
# MODEL CANDIDATES
# ─────────────────────────────────────────────────────────────

def _make_random_forest(n_estimators: int = 500):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_features='sqrt',
        max_samples=0.75,
        random_state=42,
        n_jobs=-1,
    )


def _make_xgboost(n_estimators: int = 500):
    return xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
    )


# ─────────────────────────────────────────────────────────────
# CV SCORING
# ─────────────────────────────────────────────────────────────

def _score_model(name: str, regressor, X, y_log, cat_cols):
    """Run 10-fold CV + a holdout MAE for a regressor."""
    pre  = build_preprocessor(cat_cols=cat_cols)
    pipe = Pipeline([('preprocessor', pre), ('regressor', regressor)])

    cv_scores = cross_val_score(pipe, X, y_log, cv=KFOLD, scoring='r2', n_jobs=-1)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_log, test_size=0.2, random_state=42)
    pipe.fit(X_tr, y_tr)
    y_pred_log = pipe.predict(X_te)
    mae = mean_absolute_error(np.expm1(y_te), np.expm1(y_pred_log))

    print(f"  {name:18s}  CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}  "
          f"MAE = ₹ {mae:.2f} Cr")
    return {
        'name':      name,
        'r2_mean':   cv_scores.mean(),
        'r2_std':    cv_scores.std(),
        'mae':       mae,
        'pipe':      pipe,
    }


def benchmark(X, y_log, n_estimators: int = 500):
    """Compare Random Forest vs XGBoost. Returns the winning result dict."""
    cat_cols = _get_cat_cols(X)

    print(f"\n[Benchmark] Comparing models with {n_estimators} estimators each...")
    print(f"  Features: {len(NUM_COLS)} numeric + {len(cat_cols)} categorical "
          f"({'with city' if 'city' in cat_cols else 'no city'})")
    print(f"  10-fold CV on log-transformed price\n")

    results = [
        _score_model('RandomForest',
                     _make_random_forest(n_estimators), X, y_log, cat_cols),
        _score_model('XGBoost',
                     _make_xgboost(n_estimators), X, y_log, cat_cols),
    ]

    # Winner = highest CV R²
    winner = max(results, key=lambda r: r['r2_mean'])
    print(f"\n  ✓ Winner: {winner['name']}  "
          f"(CV R² = {winner['r2_mean']:.4f}, MAE = ₹{winner['mae']:.2f} Cr)")

    return winner, results


# ─────────────────────────────────────────────────────────────
# FINAL FIT + SAVE
# ─────────────────────────────────────────────────────────────

def fit_final_pipeline(X, y, regressor):
    """Fit the chosen regressor on the full dataset (with log target)."""
    cat_cols = _get_cat_cols(X)
    pre  = build_preprocessor(cat_cols=cat_cols)
    pipe = Pipeline([('preprocessor', pre), ('regressor', regressor)])

    y_log = np.log1p(y)
    pipe.fit(X, y_log)

    train_pred = np.expm1(pipe.predict(X))
    print(f"\n  Train R²  = {r2_score(y, train_pred):.4f}")
    print(f"  Train MAE = ₹ {mean_absolute_error(y, train_pred):.2f} Cr")
    return pipe


def save_pipeline(pipe, X, pipeline_path=PIPELINE_PATH, df_path=DF_PATH):
    os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)
    with open(pipeline_path, 'wb') as f:
        pickle.dump(pipe, f)
    with open(df_path, 'wb') as f:
        pickle.dump(X, f)
    print(f"\n  Saved → {pipeline_path}")
    print(f"  Saved → {df_path}")


def load_pipeline(path: str = PIPELINE_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────
# PREDICT  (used by Streamlit app)
# ─────────────────────────────────────────────────────────────

def predict_price(
    pipe,
    property_type:  str,
    sector:         str,
    bedrooms:       int,
    bathrooms:      int,
    balcony:        str,
    age_possession: str,
    built_up_area:  float,
    servant_room:    int = 0,
    store_room:      int = 0,
    furnishing_type: str = 'unfurnished',
    luxury_category: str = 'Low',
    floor_category:  str = 'Mid Floor',
    city:            str = None,
) -> float:
    """
    Returns predicted price in Crore (₹ Cr).

    `city` is required for multi-city models. For single-city models
    pass None — the model auto-detects which column set to expect.
    """
    # Detect whether the trained pipeline expects 'city'
    expects_city = False
    try:
        ct = pipe.named_steps.get('preprocessor')
        if ct is not None:
            for name, _, cols in ct.transformers_:
                if name == 'cat' and 'city' in cols:
                    expects_city = True
                    break
    except Exception:
        pass

    if expects_city:
        if city is None:
            city = 'bangalore' if 'bangalore' in str(sector).lower() else 'gurgaon'
        row = pd.DataFrame([[
            city, property_type, sector, bedrooms, bathrooms, balcony,
            age_possession, built_up_area, servant_room, store_room,
            furnishing_type, luxury_category, floor_category,
        ]], columns=[
            'city', 'property_type', 'sector', 'bedRoom', 'bathroom',
            'balcony', 'agePossession', 'built_up_area', 'servant room',
            'store room', 'furnishing_type', 'luxury_category',
            'floor_category',
        ])
    else:
        row = pd.DataFrame([[
            property_type, sector, bedrooms, bathrooms, balcony,
            age_possession, built_up_area, servant_room, store_room,
            furnishing_type, luxury_category, floor_category,
        ]], columns=[
            'property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
            'agePossession', 'built_up_area', 'servant room', 'store room',
            'furnishing_type', 'luxury_category', 'floor_category',
        ])

    return round(float(np.expm1(pipe.predict(row))[0]), 2)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='Use 100 trees instead of 500 (much faster, ~2%% R2 lower)')
    ap.add_argument('--skip-cv', action='store_true',
                    help='Skip the benchmark - just train Random Forest with defaults')
    ap.add_argument('--model', choices=['rf', 'xgb', 'auto'], default='auto',
                    help="Force a specific model (default: 'auto' = pick winner)")
    args = ap.parse_args()

    n_estimators = 100 if args.quick else 500

    print("=" * 60)
    print("  REAL ESTATE INTELLIGENCE — Model Training")
    print("=" * 60)

    X, y, df = load_data()
    y_log = np.log1p(y)
    print(f"\n  Data: {df.shape}  ({len(_get_cat_cols(X))} cat + "
          f"{len(NUM_COLS)} num features)")

    if args.skip_cv or args.model != 'auto':
        # Just train the chosen model
        choice = 'rf' if args.model in ('rf', 'auto') else 'xgb'
        regressor = (_make_random_forest(n_estimators) if choice == 'rf'
                     else _make_xgboost(n_estimators))
        print(f"\n[Train] {choice.upper()} on full data...")
        pipe = fit_final_pipeline(X, y, regressor)
    else:
        # Benchmark RF vs XGBoost, then train winner
        winner, results = benchmark(X, y_log, n_estimators=n_estimators)
        print(f"\n[Train] Fitting {winner['name']} on full data...")
        winning_regressor = (_make_random_forest(n_estimators)
                             if winner['name'] == 'RandomForest'
                             else _make_xgboost(n_estimators))
        pipe = fit_final_pipeline(X, y, winning_regressor)

    save_pipeline(pipe, X)

    # Sanity-check predictions
    print(f"\n[Sanity Check] Sample predictions:")
    samples = [
        ('flat',  'whitefield',     'bangalore'),
        ('flat',  'sector 57',      'gurgaon'),
        ('house', 'hebbal',         'bangalore'),
        ('house', 'sector 102',     'gurgaon'),
    ]
    for ptype, sec, city in samples:
        p = predict_price(
            pipe, ptype, sec, 3, 2, '2', 'Relatively New', 1500,
            furnishing_type='semifurnished', luxury_category='Medium',
            floor_category='Mid Floor', city=city,
        )
        print(f"  {city:10s} {ptype:5s} 3 BHK 1500 sqft in {sec:18s} → ₹ {p} Cr")

    print(f"\n✓ Done. Pipeline saved to {PIPELINE_PATH}")
    print(f"  Next: streamlit run app/app.py")
