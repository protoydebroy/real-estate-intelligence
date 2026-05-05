# ============================================================
# recommender.py  —  Multi-City Property Similarity Engine
# ============================================================
# Builds three cosine similarity matrices over the model-ready dataset
# (gurgaon + bangalore combined). Properties are similar based on:
#
#   1. Numerical profile (price, area, bedrooms, bathrooms)
#   2. Categorical profile (luxury, furnishing, age, type)
#   3. Location (sector / neighborhood)
#
# Combined: w1*sim_num + w2*sim_cat + w3*sim_loc
#
# Each property has a derived 'city' field (gurgaon or bangalore)
# detected from the sector name.
#
# USAGE:
#   from recommender import PropertyRecommender
#   rec = PropertyRecommender('all_final_v2.csv')
#   rec.recommend_by_filters(city='bangalore', sector='whitefield, bangalore',
#                            top_n=5)
# ============================================================

import re
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_data(path='all_final_v2.csv') -> pd.DataFrame:
    """Load model-ready CSV. Uses explicit 'city' column if present,
    otherwise derives it from the sector name."""
    df = pd.read_csv(path).reset_index(drop=True)
    if 'city' not in df.columns:
        df['city'] = df['sector'].apply(_detect_city)
    return df


def _detect_city(sector: str) -> str:
    """Fallback: infer city from the sector name when 'city' column missing."""
    s = str(sector).lower()
    if 'bangalore' in s:
        return 'bangalore'
    return 'gurgaon'


# ─────────────────────────────────────────────────────────────
# SIMILARITY MODELS
# ─────────────────────────────────────────────────────────────

def _build_numerical_sim(df: pd.DataFrame) -> np.ndarray:
    """Cosine similarity over scaled numerical features."""
    cols = ['price', 'bedRoom', 'bathroom', 'built_up_area',
            'servant room', 'store room']
    cols = [c for c in cols if c in df.columns]
    X = StandardScaler().fit_transform(df[cols].fillna(0))
    return cosine_similarity(X)


def _build_categorical_sim(df: pd.DataFrame) -> np.ndarray:
    """Cosine similarity over one-hot encoded categoricals."""
    cols = ['property_type', 'agePossession', 'furnishing_type',
            'luxury_category', 'floor_category', 'balcony']
    cols = [c for c in cols if c in df.columns]
    X = pd.get_dummies(df[cols].astype(str), drop_first=False).values
    return cosine_similarity(X.astype(float))


def _build_location_sim(df: pd.DataFrame) -> np.ndarray:
    """Cosine similarity over one-hot encoded sector + city."""
    X = pd.get_dummies(df[['sector', 'city']].astype(str)).values
    return cosine_similarity(X.astype(float))


# ─────────────────────────────────────────────────────────────
# RECOMMENDER CLASS
# ─────────────────────────────────────────────────────────────

class PropertyRecommender:
    """
    Multi-signal property recommender across Gurgaon + Bangalore.

    Default weights (matching the original Gurgaon recommender):
      numerical (price/area):  30
      categorical (type/lux):  20
      location (sector/city):   8
    """

    def __init__(
        self,
        csv_path: str = 'all_final_v2.csv',
        w_numerical: float = 30,
        w_categorical: float = 20,
        w_location: float = 8,
    ):
        print("Loading data...")
        self.df = load_data(csv_path)

        print("Building numerical similarity (price / area / rooms)...")
        self.sim_num = _build_numerical_sim(self.df)

        print("Building categorical similarity (type / luxury / furnishing)...")
        self.sim_cat = _build_categorical_sim(self.df)

        print("Building location similarity (sector / city)...")
        self.sim_loc = _build_location_sim(self.df)

        self.w_num = w_numerical
        self.w_cat = w_categorical
        self.w_loc = w_location

        print(f"Ready — {len(self.df)} properties loaded.")

    def _combined(self) -> np.ndarray:
        return (self.w_num * self.sim_num
                + self.w_cat * self.sim_cat
                + self.w_loc * self.sim_loc)

    def cities(self) -> list:
        return sorted(self.df['city'].unique().tolist())

    def sectors(self, city: str = None) -> list:
        d = self.df if city is None else self.df[self.df['city'] == city]
        return sorted(d['sector'].unique().tolist())

    def update_weights(
        self,
        w_numerical: float = None,
        w_categorical: float = None,
        w_location: float = None,
    ):
        if w_numerical    is not None: self.w_num = w_numerical
        if w_categorical  is not None: self.w_cat = w_categorical
        if w_location     is not None: self.w_loc = w_location

    # ── Recommendation by index ───────────────────────────────

    def recommend(self, idx: int, top_n: int = 5,
                  same_city_only: bool = False) -> pd.DataFrame:
        """
        Return top_n similar properties to the property at row `idx`.
        """
        if idx < 0 or idx >= len(self.df):
            raise ValueError(f"Index {idx} out of range")

        scores = self._combined()[idx]
        order  = np.argsort(scores)[::-1]
        order  = [i for i in order if i != idx]   # drop self

        if same_city_only:
            ref_city = self.df.iloc[idx]['city']
            order = [i for i in order if self.df.iloc[i]['city'] == ref_city]

        top = order[:top_n]
        out = self.df.iloc[top].copy().reset_index(drop=True)
        out['SimilarityScore'] = [round(float(scores[i]), 4) for i in top]
        out.insert(0, 'Rank', range(1, len(out) + 1))
        return out

    # ── Recommendation by filters (used by the app) ───────────

    def recommend_by_filters(
        self,
        city: str = None,
        sector: str = None,
        property_type: str = None,
        bedrooms: int = None,
        budget_max: float = None,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Find a reference property matching the filters, then return
        its top_n nearest neighbours.
        """
        candidates = self.df.copy()
        if city:
            candidates = candidates[candidates['city'] == city]
        if sector:
            candidates = candidates[candidates['sector'] == sector]
        if property_type:
            candidates = candidates[candidates['property_type'] == property_type]
        if bedrooms:
            candidates = candidates[candidates['bedRoom'] == bedrooms]
        if budget_max:
            candidates = candidates[candidates['price'] <= budget_max]

        if candidates.empty:
            return pd.DataFrame()

        # Use the median-priced match as the reference
        ref_idx = candidates.iloc[
            (candidates['price'] - candidates['price'].median()).abs().argsort()
        ].index[0]

        return self.recommend(ref_idx, top_n=top_n,
                              same_city_only=bool(city))


if __name__ == '__main__':
    rec = PropertyRecommender('all_final_v2.csv')

    print("\n── Bangalore: 3 BHK flat in Whitefield ──")
    print(rec.recommend_by_filters(
        city='bangalore', property_type='flat', bedrooms=3,
        budget_max=3.0, top_n=5).to_string(index=False))

    print("\n── Gurgaon: 4 BHK house ──")
    print(rec.recommend_by_filters(
        city='gurgaon', property_type='house', bedrooms=4,
        top_n=5).to_string(index=False))
