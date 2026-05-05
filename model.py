# ============================================================
# model.py  —  Gurgaon Property Price Prediction
# ============================================================
# INPUT:  gurgaon_properties_post_feature_selection_v2.csv
#         Columns: property_type, sector, price, bedRoom, bathroom,
#                  balcony, agePossession, built_up_area, servant room,
#                  store room, furnishing_type, luxury_category, floor_category
#
# OUTPUT: pipeline.pkl  (trained sklearn Pipeline)
#         df.pkl        (feature DataFrame X — for option lookup)
#
# USAGE:
#   python model.py                    → train + save
#   from model import load_pipeline, predict_price
#
# INSTALL: pip install pandas numpy scikit-learn xgboost
# ============================================================

import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, AdaBoostRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

DATA_PATH     = 'gurgaon_properties_post_feature_selection_v2.csv'
PIPELINE_PATH = 'pipeline.pkl'
DF_PATH       = 'df.pkl'
KFOLD         = KFold(n_splits=10, shuffle=True, random_state=42)

# Exact columns from the v2 CSV
NUM_COLS = ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']
# 'city' is included if present in the dataset (multi-city setup)
# Falls back gracefully to single-city when city column is absent.
CAT_COLS_BASE = ['property_type', 'sector', 'balcony', 'agePossession',
                 'furnishing_type', 'luxury_category', 'floor_category']


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_data(path=DATA_PATH):
    """
    Loads the v2 feature-selected CSV.
    furnishing_type: replaces 0/1/2 floats with string labels.
    Returns X, y, full df.
    """
    df = pd.read_csv(path)

    # Map numeric furnishing codes → readable strings
    df['furnishing_type'] = df['furnishing_type'].replace(
        {0.0: 'unfurnished', 1.0: 'semifurnished', 2.0: 'furnished'})

    X = df.drop(columns=['price'])
    y = df['price']
    return X, y, df


# ─────────────────────────────────────────────────────────────
# PREPROCESSOR
# ─────────────────────────────────────────────────────────────

def _get_cat_cols(X) -> list:
    """Return the categorical columns present in X.
    Includes 'city' if available (multi-city dataset)."""
    cols = ['city'] + CAT_COLS_BASE if 'city' in X.columns else CAT_COLS_BASE
    return [c for c in cols if c in X.columns]


def build_preprocessor(cat_cols=None):
    """
    ColumnTransformer:
      NUM_COLS  → StandardScaler
      cat_cols  → OneHotEncoder(drop='first', handle_unknown='ignore')

    If cat_cols is None, uses CAT_COLS_BASE (no city). For the multi-city
    dataset, pass cat_cols=_get_cat_cols(X) so 'city' is included.
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
# SINGLE MODEL SCORER
# ─────────────────────────────────────────────────────────────

def _score(name, model, X, y_log):
    preprocessor = build_preprocessor(cat_cols=_get_cat_cols(X))
    pipe = Pipeline([('preprocessor', preprocessor), ('regressor', model)])

    cv_scores = cross_val_score(pipe, X, y_log, cv=KFOLD, scoring='r2')

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_log, test_size=0.2, random_state=42)
    pipe.fit(X_tr, y_tr)
    mae = mean_absolute_error(np.expm1(y_te), np.expm1(pipe.predict(X_te)))

    return {'name': name, 'r2': round(cv_scores.mean(), 4),
            'r2_std': round(cv_scores.std(), 4), 'mae_cr': round(mae, 4)}


# ─────────────────────────────────────────────────────────────
# BASELINE
# ─────────────────────────────────────────────────────────────

def baseline(X, y):
    """SVR(rbf) baseline — replicates baseline_model.ipynb."""
    print("\n[Baseline] SVR RBF kernel (10-fold CV)...")
    y_log = np.log1p(y)
    result = _score('SVR', SVR(kernel='rbf'), X, y_log)
    print(f"  R² = {result['r2']} ± {result['r2_std']}   MAE = {result['mae_cr']} Cr")
    return result


# ─────────────────────────────────────────────────────────────
# MODEL COMPARISON
# ─────────────────────────────────────────────────────────────

def compare_models(X, y) -> pd.DataFrame:
    """
    Evaluates 11 algorithms (replicates model-selection.ipynb).
    Returns DataFrame sorted by MAE ascending.
    """
    print("\n[Model Comparison] 11 algorithms × 10-fold CV...")
    y_log = np.log1p(y)

    models = {
        'LinearRegression':  LinearRegression(),
        'Ridge':             Ridge(),
        'Lasso':             Lasso(),
        'SVR':               SVR(),
        'DecisionTree':      DecisionTreeRegressor(),
        'RandomForest':      RandomForestRegressor(random_state=42),
        'ExtraTrees':        ExtraTreesRegressor(random_state=42),
        'GradientBoosting':  GradientBoostingRegressor(),
        'AdaBoost':          AdaBoostRegressor(),
        'MLP':               MLPRegressor(max_iter=500, random_state=42),
    }
    if _HAS_XGB:
        models['XGBoost'] = XGBRegressor(verbosity=0, random_state=42)

    results = []
    for name, model in models.items():
        print(f"  {name}...", end=' ', flush=True)
        try:
            row = _score(name, model, X, y_log)
            results.append(row)
            print(f"R²={row['r2']}  MAE={row['mae_cr']} Cr")
        except Exception as e:
            print(f"FAILED ({e})")

    df_results = pd.DataFrame(results).sort_values('mae_cr').reset_index(drop=True)
    print("\n── Ranked by MAE ──")
    print(df_results.to_string(index=False))
    return df_results


# ─────────────────────────────────────────────────────────────
# HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────

def tune_random_forest(X, y) -> Pipeline:
    """
    GridSearchCV over RandomForestRegressor (replicates model-selection.ipynb).
    Returns best pipeline — not yet fit on full data.
    """
    print("\n[Tuning] GridSearchCV on RandomForest...")
    y_log = np.log1p(y)

    preprocessor = build_preprocessor(cat_cols=_get_cat_cols(X))
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42)),
    ])

    param_grid = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth':    [None, 10, 20],
        'regressor__max_samples':  [0.5, 0.75, 1.0],
        'regressor__max_features': ['sqrt', 'log2'],
    }

    search = GridSearchCV(pipe, param_grid, cv=KFOLD,
                          scoring='r2', n_jobs=-1, verbose=2)
    search.fit(X, y_log)

    print(f"\n  Best params : {search.best_params_}")
    print(f"  Best CV R²  : {search.best_score_:.4f}")
    return search.best_estimator_


# ─────────────────────────────────────────────────────────────
# FINAL PIPELINE
# ─────────────────────────────────────────────────────────────

def build_final_pipeline(X, y, n_estimators=500) -> Pipeline:
    """
    RandomForest(500 trees) trained on FULL dataset.
    Matches the final pipeline saved in model-selection.ipynb.
    Returns fitted pipeline.
    """
    print(f"\n[Final] RandomForest({n_estimators} trees) — full data...")
    y_log = np.log1p(y)

    preprocessor = build_preprocessor(cat_cols=_get_cat_cols(X))
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=n_estimators, random_state=42)),
    ])

    # Cross-validate before fitting on full data
    cv = cross_val_score(pipe, X, y_log, cv=KFOLD, scoring='r2')
    print(f"  CV R² = {cv.mean():.4f} ± {cv.std():.4f}")

    pipe.fit(X, y_log)
    print("  Fitted on full dataset.")
    return pipe


# ─────────────────────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────────────────────

def save_pipeline(pipe, X, pipeline_path=PIPELINE_PATH, df_path=DF_PATH):
    with open(pipeline_path, 'wb') as f:
        pickle.dump(pipe, f)
    with open(df_path, 'wb') as f:
        pickle.dump(X, f)
    print(f"\n  Saved → {pipeline_path}  and  {df_path}")


def load_pipeline(pipeline_path=PIPELINE_PATH):
    with open(pipeline_path, 'rb') as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────

def predict_price(
    pipe,
    property_type: str,       # 'flat' | 'house'
    sector: str,              # e.g. 'sector 57'
    bedrooms: int,
    bathrooms: int,
    balcony: str,             # '0' | '1' | '2' | '3+'
    age_possession: str,      # 'New Property' | 'Relatively New' | ...
    built_up_area: float,     # sqft
    servant_room: int = 0,
    store_room: int = 0,
    furnishing_type: str = 'unfurnished',
    luxury_category: str = 'Low',
    floor_category: str = 'Mid Floor',
    city: str = None,         # 'gurgaon' | 'bangalore' (multi-city only)
) -> float:
    """
    Returns predicted price in Crore (₹ Cr).

    The `city` parameter is required when the model was trained on the
    multi-city dataset (all_final_v2.csv with 'city' column). For a
    single-city Gurgaon-only model it can be left as None.

    Example
    -------
    >>> pipe = load_pipeline()
    >>> predict_price(pipe, 'flat', 'whitefield, bangalore', 3, 2, '2',
    ...               'Relatively New', 1500, city='bangalore')
    1.83
    """
    # Detect if pipeline expects 'city' from preprocessor cat columns
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
            # Auto-detect from sector if not provided
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


def get_valid_options(df_path=DF_PATH):
    """Print all valid categorical values for prediction inputs."""
    with open(df_path, 'rb') as f:
        X = pickle.load(f)
    for col in ['property_type', 'sector', 'balcony', 'agePossession',
                'furnishing_type', 'luxury_category', 'floor_category']:
        vals = sorted(X[col].dropna().unique().tolist())
        print(f"\n{col}:\n  {vals}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Load data
    X, y, df = load_data()
    print(f"Loaded: {df.shape}  |  Sectors: {X['sector'].nunique()}")
    print(f"Price range: ₹{y.min()} – ₹{y.max()} Cr\n")

    # Baseline
    baseline(X, y)

    # Compare all models (comment out if slow ~5-10 min)
    # compare_models(X, y)

    # Build and save final pipeline
    pipe = build_final_pipeline(X, y, n_estimators=500)
    save_pipeline(pipe, X)

    # Sample predictions
    print("\n── Sample Predictions ──")
    tests = [
        dict(property_type='flat',  sector='sector 57',  bedrooms=3, bathrooms=2,
             balcony='2', age_possession='Relatively New', built_up_area=1500,
             furnishing_type='semifurnished', luxury_category='Medium',
             floor_category='Mid Floor'),
        dict(property_type='house', sector='sector 102', bedrooms=4, bathrooms=3,
             balcony='3+', age_possession='New Property', built_up_area=2750,
             servant_room=1, furnishing_type='unfurnished', luxury_category='Low',
             floor_category='Low Floor'),
        dict(property_type='flat',  sector='sector 28',  bedrooms=2, bathrooms=2,
             balcony='1', age_possession='Old Property', built_up_area=900,
             furnishing_type='furnished', luxury_category='Low',
             floor_category='High Floor'),
    ]
    for t in tests:
        price = predict_price(pipe, **t)
        print(f"  {t['bedrooms']} BHK {t['property_type']} in {t['sector']} "
              f"({t['built_up_area']} sqft) → ₹ {price} Cr")
