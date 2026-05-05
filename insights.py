# ============================================================
# insights.py  —  Feature Impact Analysis
# ============================================================
# Direct port of insights-module.ipynb with multi-city support.
#
# Methodology (matches notebook line-for-line):
#   1. Drop low-importance features: store room, floor_category, balcony
#   2. agePossession  → 3 buckets: 'new', 'old', 'under construction'
#   3. property_type  → {flat: 0, house: 1}              (ordinal)
#   4. luxury_category → {Low: 0, Medium: 1, High: 2}    (ordinal)
#   5. furnishing_type → already 0/1/2                   (ordinal)
#   6. sector + agePossession → pd.get_dummies(drop_first=True)
#   7. y_log = np.log1p(price)
#   8. X_scaled = StandardScaler.fit_transform(X)
#   9. Ridge(alpha=0.0001).fit(X_scaled, y_log)
#  10. OLS via statsmodels for full summary (R², p-values, conf intervals)
#
# Notebook reports:
#   - 10-fold CV R² ≈ 0.85 (LinearRegression on the same X_scaled, y_log)
#   - OLS R² ≈ 0.865, Adj R² ≈ 0.860
#
# Multi-city extension: when 'city' is in the dataframe, this module
# fits one model PER city so each city's drivers are reported separately
# (Bangalore neighborhoods don't pollute Gurgaon analysis and vice versa).
#
# USAGE:
#   from insights import InsightsAnalyzer
#   ins = InsightsAnalyzer('all_final_v2.csv')
#   ins.coef_df(city='gurgaon')                   # full coef table
#   ins.ols_summary(city='gurgaon')               # statsmodels summary
#   ins.cv_score(city='gurgaon')                  # 10-fold CV R²
#   ins.feature_importance(city='gurgaon')        # ranked by |coef|
#   ins.predict_impact('gurgaon', 'built_up_area', delta=100)
#   ins.city_comparison()
# ============================================================

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score

try:
    import statsmodels.api as sm
    _HAS_SM = True
except ImportError:
    _HAS_SM = False


# ─────────────────────────────────────────────────────────────
# CONFIG  (matches notebook)
# ─────────────────────────────────────────────────────────────

DROP_COLS = ['store room', 'floor_category', 'balcony']

AGE_MAP = {
    'Relatively New':       'new',
    'New Property':         'new',
    'Moderately Old':       'old',
    'Old Property':         'old',
    'Under Construction':   'under construction',
}
PROPERTY_TYPE_MAP   = {'flat': 0, 'house': 1}
LUXURY_MAP          = {'Low': 0, 'Medium': 1, 'High': 2}

OHE_COLS            = ['sector', 'agePossession']

KFOLD = KFold(n_splits=10, shuffle=True, random_state=42)


# ─────────────────────────────────────────────────────────────
# DATA PREP  (notebook cells 1-15 condensed)
# ─────────────────────────────────────────────────────────────

def _detect_city(sector: str) -> str:
    return 'bangalore' if 'bangalore' in str(sector).lower() else 'gurgaon'


def _prepare(df: pd.DataFrame) -> tuple:
    """
    Replicates the notebook preprocessing exactly.
    Returns (X_scaled, y_log, scaler, X_unscaled).
    """
    df = df.copy()

    # Step 1: drop low-impact cols
    df = df.drop(columns=DROP_COLS, errors='ignore')

    # Step 2: agePossession → 3 buckets
    df['agePossession'] = df['agePossession'].replace(AGE_MAP)

    # Step 3: ordinal encoding
    df['property_type']   = df['property_type'].replace(PROPERTY_TYPE_MAP)
    df['luxury_category'] = df['luxury_category'].replace(LUXURY_MAP)

    # furnishing_type may be string ('unfurnished'/'semi'/'furnished')
    # or numeric (0/1/2). Convert strings back to 0/1/2 if needed.
    if df['furnishing_type'].dtype == object:
        df['furnishing_type'] = df['furnishing_type'].replace({
            'unfurnished':   0,
            'semifurnished': 1,
            'furnished':     2,
        })

    # Drop city if present (we operate per-city)
    df = df.drop(columns=['city'], errors='ignore')

    # Step 4: OHE sector + agePossession
    df = pd.get_dummies(df, columns=OHE_COLS, drop_first=True)

    # Coerce all to float (get_dummies returns bool which scaler hates)
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(float)

    # Separate X / y, log-transform target
    X       = df.drop(columns=['price']).astype(float)
    y_log   = np.log1p(df['price'])

    # Standardise X
    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns, index=X.index,
    )

    return X_scaled, y_log, scaler, X


# ─────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────

class InsightsAnalyzer:
    """
    Per-city Ridge regression for feature-impact insights,
    matching insights-module.ipynb.

    One model per city is fit on demand and cached.
    """

    def __init__(self, csv_path: str = 'all_final_v2.csv'):
        df = pd.read_csv(csv_path)
        if 'city' not in df.columns:
            df['city'] = df['sector'].apply(_detect_city)
        self.df    = df
        self._cache = {}   # {city_key: (ridge, lr, X_scaled, y_log, scaler, X_unscaled)}

    # ── basic accessors ──────────────────────────────────────

    def cities(self) -> list:
        return sorted(self.df['city'].unique().tolist())

    def _slice(self, city: str = None) -> pd.DataFrame:
        if city is None:
            return self.df.drop(columns=['city'], errors='ignore')
        return self.df[self.df['city'] == city].drop(columns=['city'])

    # ── model fitting ────────────────────────────────────────

    def _fit(self, city: str = None):
        """Fit Ridge + LinearRegression for the given city (cached)."""
        key = city or '__all__'
        if key in self._cache:
            return self._cache[key]

        sub = self._slice(city)
        X_scaled, y_log, scaler, X_unscaled = _prepare(sub)

        # Notebook fits BOTH LinearRegression and Ridge(alpha=0.0001).
        # We do the same so we can report either set of coefficients.
        lr    = LinearRegression()
        ridge = Ridge(alpha=0.0001)
        lr.fit(X_scaled, y_log)
        ridge.fit(X_scaled, y_log)

        result = (ridge, lr, X_scaled, y_log, scaler, X_unscaled)
        self._cache[key] = result
        return result

    # ── 10-fold CV R²  (notebook cells 18-19) ────────────────

    def cv_score(self, city: str = None) -> dict:
        """
        Returns mean and std R² over 10 folds, using LinearRegression
        on the scaled features — same as the notebook.
        """
        sub = self._slice(city)
        X_scaled, y_log, _, _ = _prepare(sub)

        scores = cross_val_score(
            LinearRegression(), X_scaled, y_log,
            cv=KFOLD, scoring='r2',
        )
        return {
            'r2_mean':    float(scores.mean()),
            'r2_std':     float(scores.std()),
            'n_features': X_scaled.shape[1],
            'n_samples':  X_scaled.shape[0],
        }

    # ── coef_df  (notebook cells 23-24) ──────────────────────

    def coef_df(self, city: str = None) -> pd.DataFrame:
        """
        Notebook's coef_df: flat (feature, coef) DataFrame from Ridge.
        """
        ridge, _, X_scaled, _, _, _ = self._fit(city)
        return pd.DataFrame({
            'feature': X_scaled.columns,
            'coef':    ridge.coef_,
        })

    # ── OLS summary  (notebook cell 25) ──────────────────────

    def ols_summary(self, city: str = None):
        """
        Returns the full statsmodels OLS summary with p-values,
        confidence intervals and R²/Adj R². Matches notebook cell 25.
        """
        if not _HAS_SM:
            raise ImportError(
                "statsmodels not installed. Run: pip install statsmodels")

        _, _, X_scaled, y_log, _, _ = self._fit(city)
        X_with_const = sm.add_constant(X_scaled)
        model = sm.OLS(y_log, X_with_const).fit()
        return model

    def ols_metrics(self, city: str = None) -> dict:
        """Just the headline OLS metrics — R², Adj R², n_obs."""
        m = self.ols_summary(city)
        return {
            'r2':         float(m.rsquared),
            'adj_r2':     float(m.rsquared_adj),
            'fstat':      float(m.fvalue),
            'aic':        float(m.aic),
            'n_obs':      int(m.nobs),
            'df_model':   int(m.df_model),
        }

    # ── ranked feature importance ────────────────────────────

    def feature_importance(self, city: str = None,
                           top_n: int = 20) -> pd.DataFrame:
        """
        Coefficients ranked by absolute magnitude. Each row shows:
          feature, coefficient (in log-price per σ), abs_coef, direction.
        """
        coef = self.coef_df(city)
        coef['abs_coef']  = coef['coef'].abs()
        coef['direction'] = coef['coef'].apply(lambda c: '+' if c > 0 else '-')
        coef = coef.sort_values('abs_coef', ascending=False).reset_index(drop=True)
        return coef.head(top_n).rename(columns={'coef': 'coefficient'})

    # ── plain-English impact translator ──────────────────────

    def predict_impact(self, city: str, feature: str,
                       delta: float = 1.0) -> dict:
        """
        For a numeric feature, estimate how much price changes (% and
        absolute log-change) if you increase that feature by `delta`
        raw units (e.g. 100 sqft of built_up_area).

        Math (matches notebook bottom cells):
          coef        = log-price change per 1 standardised unit
          delta_std   = delta_raw / scaler.scale_[col]
          log_change  = coef × delta_std
          pct_change  = (np.expm1(log_change) - 1) × 100
        """
        ridge, _, X_scaled, _, scaler, X_unscaled = self._fit(city)
        if feature not in X_scaled.columns:
            raise ValueError(
                f"Feature {feature!r} not found.\n"
                f"Available numeric: {[c for c in X_scaled.columns if not c.startswith(('sector_','agePossession_'))][:15]}"
            )

        col_idx   = list(X_scaled.columns).index(feature)
        coef      = float(ridge.coef_[col_idx])

        # In the notebook X is scaled, so 1 unit in scaled space = std in raw space.
        # delta_std (in scaled units) = delta_raw / std_raw
        std_raw   = float(scaler.scale_[col_idx])
        delta_std = delta / std_raw if std_raw else delta

        log_change = coef * delta_std
        pct_change = float(np.expm1(log_change) * 100)

        return {
            'feature':      feature,
            'delta_raw':    float(delta),
            'std_raw':      round(std_raw, 4),
            'coefficient':  round(coef, 6),
            'log_change':   round(log_change, 6),
            'price_pct':    round(pct_change, 4),
        }

    # ── high-level summary helpers ───────────────────────────

    def top_drivers(self, city: str = None, n: int = 10) -> dict:
        """Most positive and most negative coefficients."""
        coef = self.coef_df(city).copy()
        coef['abs_coef'] = coef['coef'].abs()
        coef = coef.sort_values('coef', ascending=False)
        return {
            'positive': coef.head(n).reset_index(drop=True),
            'negative': coef.tail(n).iloc[::-1].reset_index(drop=True),
        }

    def city_comparison(self) -> pd.DataFrame:
        """Side-by-side stats for both cities."""
        rows = []
        for city in self.cities():
            sub = self.df[self.df['city'] == city]
            rows.append({
                'city':              city,
                'n_properties':      len(sub),
                'flats':             int((sub['property_type'] == 'flat').sum()),
                'houses':            int((sub['property_type'] == 'house').sum()),
                'price_min_cr':      round(sub['price'].min(), 2),
                'price_median_cr':   round(sub['price'].median(), 2),
                'price_max_cr':      round(sub['price'].max(), 2),
                'area_median_sqft':  int(sub['built_up_area'].median()),
                'unique_locations':  int(sub['sector'].nunique()),
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ins = InsightsAnalyzer('all_final_v2.csv')

    print("── City comparison ──")
    print(ins.city_comparison().to_string(index=False))

    for city in ins.cities():
        print(f"\n{'='*60}\n  {city.upper()}\n{'='*60}")

        cv = ins.cv_score(city)
        print(f"\n10-fold CV R² (LinearRegression): "
              f"{cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}  "
              f"({cv['n_samples']} rows × {cv['n_features']} features)")

        if _HAS_SM:
            ols = ins.ols_metrics(city)
            print(f"OLS R² = {ols['r2']:.4f}   Adj R² = {ols['adj_r2']:.4f}   "
                  f"F-stat = {ols['fstat']:.1f}")

        print(f"\nTop 8 drivers (|coef|):")
        print(ins.feature_importance(city, top_n=8).to_string(index=False))

        print(f"\nImpact: +100 sqft built_up_area in {city}:")
        impact = ins.predict_impact(city, 'built_up_area', 100)
        print(f"  price_pct = {impact['price_pct']:+.2f}%   "
              f"(coef={impact['coefficient']}, std={impact['std_raw']})")
