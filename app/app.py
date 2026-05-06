# ============================================================
# app.py  —  Real Estate Intelligence App
# ============================================================
# Sidebar navigation with three pages:
#   1. Price Predictor → ML model predicts property price
#   2. Recommender     → Find similar properties
#   3. Insights        → Feature impact analysis (Ridge regression)
#
# Both Gurgaon and Bangalore supported across all three pages.
#
# RUN:
#   pip install streamlit pandas scikit-learn numpy
#   streamlit run app.py
#
# REQUIRED FILES (same folder):
#   - all_final_v2.csv     (combined model-ready dataset, with 'city' col)
#   - pipeline.pkl         (trained model)
#   - df.pkl               (feature DataFrame)
#   - model.py
#   - recommender.py
#   - insights.py
# ============================================================

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Make sibling modules importable from this folder + ../model
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, 'model'))

import streamlit as st
import pandas as pd
import numpy as np

from train_model import load_pipeline, predict_price
from recommender import PropertyRecommender
from insights import InsightsAnalyzer
from geo_coords import add_coords

# Standard file locations (relative to repo root)
DATA_PATH = os.path.join(REPO_ROOT, 'data', 'processed', 'all_final_v2.csv')
PIPELINE_PATH = os.path.join(REPO_ROOT, 'model', 'pipeline.pkl')
DF_PATH = os.path.join(REPO_ROOT, 'model', 'df.pkl')


# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Real Estate Intelligence",
    page_icon="🏙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme-safe CSS — uses transparent backgrounds + neutral borders so it
# adapts to both light and dark mode automatically.
st.markdown("""
<style>
    /* Use full available width — remove cramped right-shifted layout */
    .block-container {
        padding-top: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 100%;
    }

    /* Page title styling */
    .page-title {
        font-size: 1.9rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .page-caption {
        opacity: 0.65;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* Property card — theme-adaptive */
    .property-card {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 14px;
        background: rgba(128, 128, 128, 0.06);
    }
    .property-card .rank {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.6;
    }
    .property-card .name {
        font-size: 16px;
        font-weight: 600;
        margin: 6px 0;
    }
    .property-card .meta {
        font-size: 13px;
        line-height: 1.7;
        opacity: 0.85;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        min-width: 240px;
        max-width: 270px;
    }
    [data-testid="stSidebar"] .stRadio > div {
        gap: 4px;
    }
    [data-testid="stSidebar"] .stRadio label {
        padding: 6px 10px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CACHED LOADERS — built around the CSV (most reliable source)
# ─────────────────────────────────────────────────────────────

@st.cache_data
def load_master_df():
    """
    Master dataframe used for all city/sector filtering.
    Always has a 'city' column.
    """
    if not os.path.isfile(DATA_PATH):
        st.error("`all_final_v2.csv` not found. Run `python preprocessing/run_all.py` first.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    if 'city' not in df.columns:
        # Defensive: derive city from sector if not present
        df['city'] = df['sector'].apply(
            lambda s: 'bangalore' if 'bangalore' in str(s).lower()
            else 'gurgaon')
    return df


@st.cache_resource(show_spinner=False)
def get_pipeline():
    """
    Load the trained pipeline. If pipeline.pkl is missing (e.g. on first
    deploy to Streamlit Cloud), train it from the CSV on the fly. This
    only runs once per cold start.
    """
    if os.path.isfile(PIPELINE_PATH):
        with st.spinner("Loading model…"):
            return load_pipeline(PIPELINE_PATH)

    # First run on a fresh deploy → train the model now
    if not os.path.isfile(DATA_PATH):
        st.error(
            f"Neither `{os.path.basename(PIPELINE_PATH)}` nor "
            f"`{os.path.basename(DATA_PATH)}` found. "
            "The repo is missing required files."
        )
        st.stop()

    from train_model import (
        load_data, _make_xgboost, fit_final_pipeline, save_pipeline,
    )
    with st.spinner("Training model on first run (≈ 1–2 minutes)…"):
        X, y, _ = load_data(DATA_PATH)
        pipe = fit_final_pipeline(X, y, _make_xgboost(n_estimators=500))
        save_pipeline(pipe, X)
    return pipe


@st.cache_resource(show_spinner="Building recommender…")
def get_recommender():
    return PropertyRecommender(DATA_PATH)


@st.cache_resource(show_spinner="Loading insights…")
def get_insights():
    return InsightsAnalyzer(DATA_PATH)


def get_sectors_for_city(city: str) -> list:
    """Return alphabetically-sorted unique sectors for a given city."""
    df = load_master_df()
    return sorted(df[df['city'] == city]['sector'].unique().tolist())


def _humanize_feature(f: str) -> str:
    if f.startswith('sector_'):
        return f.replace('sector_', '📍 ')
    if f.startswith('agePossession_'):
        return f.replace('agePossession_', '🕒 Age: ')
    if f.startswith('city_'):
        return f.replace('city_', '🏙 City: ')
    return f


# ─────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🏙 Real Estate")
    st.caption("Intelligence for Gurgaon & Bangalore")
    st.divider()

    page = st.radio(
        "Page",
        ["💰 Price Predictor", "🔍 Recommender", "📊 Insights"],
        label_visibility="collapsed",
    )

    st.divider()

    # Quick dataset summary
    df_master = load_master_df()
    st.markdown("##### Dataset")
    st.caption(f"**{len(df_master):,}** properties")
    for c in sorted(df_master['city'].unique()):
        n = (df_master['city'] == c).sum()
        st.caption(f"• {c.title()}: {n:,}")


# ─────────────────────────────────────────────────────────────
# PAGE 1 — PRICE PREDICTOR
# ─────────────────────────────────────────────────────────────

if page == "💰 Price Predictor":
    st.markdown('<div class="page-title">Predict property price</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-caption">Enter property details to get an '
                'estimated market price based on the trained ML model.</div>',
                unsafe_allow_html=True)

    pipe = get_pipeline()

    # ── City + property type
    available_cities = sorted(df_master['city'].unique().tolist())

    col_city, col_type = st.columns(2)
    with col_city:
        city = st.selectbox("City", available_cities, key="pp_city")
    with col_type:
        property_type = st.selectbox("Property type", ["flat", "house"],
                                     key="pp_type")

    # ── FILTER LOCATIONS USING THE CSV (always reliable)
    # This filters from all_final_v2.csv directly so it never falls
    # back to Gurgaon when Bangalore is picked.
    city_sectors = get_sectors_for_city(city)

    if not city_sectors:
        st.error(f"No locations found for {city}. Check your "
                 f"`all_final_v2.csv` includes {city} data.")
        st.stop()

    sector = st.selectbox(
        f"Location in {city.title()} ({len(city_sectors)} options)",
        city_sectors,
        key=f"pp_sector_{city}",   # key includes city so it resets on change
    )

    # ── Numerical inputs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        bedrooms      = st.number_input("Bedrooms", 1, 10, 3, key="pp_bed")
    with c2:
        bathrooms     = st.number_input("Bathrooms", 1, 10, 2, key="pp_bath")
    with c3:
        built_up_area = st.number_input("Built-up area (sqft)",
                                        100, 15000, 1500, step=50,
                                        key="pp_area")
    with c4:
        balcony       = st.selectbox("Balcony",
                                     ["0", "1", "2", "3", "3+"],
                                     index=2, key="pp_bal")

    # ── Categorical inputs
    c5, c6, c7 = st.columns(3)
    with c5:
        age_possession  = st.selectbox(
            "Age / Possession",
            ["New Property", "Relatively New", "Moderately Old",
             "Old Property", "Under Construction"],
            index=1, key="pp_age",
        )
    with c6:
        furnishing_type = st.selectbox(
            "Furnishing",
            ["unfurnished", "semifurnished", "furnished"],
            index=1, key="pp_furn",
        )
    with c7:
        floor_category  = st.selectbox(
            "Floor",
            ["Low Floor", "Mid Floor", "High Floor"],
            index=1, key="pp_floor",
        )

    c8, c9, c10 = st.columns(3)
    with c8:
        luxury_category = st.selectbox(
            "Luxury level", ["Low", "Medium", "High"],
            index=0, key="pp_lux",
        )
    with c9:
        servant_room    = st.selectbox("Servant room", [0, 1],
                                       key="pp_serv")
    with c10:
        store_room      = st.selectbox("Store room", [0, 1],
                                       key="pp_store")

    st.write("")
    if st.button("Predict price", type="primary", use_container_width=True):
        try:
            price = predict_price(
                pipe,
                property_type=property_type,
                sector=sector,
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                balcony=balcony,
                age_possession=age_possession,
                built_up_area=built_up_area,
                servant_room=servant_room,
                store_room=store_room,
                furnishing_type=furnishing_type,
                luxury_category=luxury_category,
                floor_category=floor_category,
                city=city,
            )

            lo, hi = round(price * 0.90, 2), round(price * 1.10, 2)
            psqft  = int(price * 10_000_000 / built_up_area)

            st.write("")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Predicted price",       f"₹ {price} Cr")
            mc2.metric("Likely range",          f"₹ {lo} – {hi} Cr")
            mc3.metric("Implied price / sqft",  f"₹ {psqft:,}")

            # Compare with similar listings
            similar = df_master[
                (df_master['city'] == city) &
                (df_master['property_type'] == property_type) &
                (df_master['bedRoom'] == bedrooms)
            ]
            if not similar.empty:
                with st.expander(
                    f"Compare with {len(similar)} similar listings "
                    f"({city.title()}, {property_type}, {bedrooms} BHK)",
                    expanded=True,
                ):
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Market median", f"₹ {similar['price'].median():.2f} Cr")
                    s2.metric("Market low",    f"₹ {similar['price'].min():.2f} Cr")
                    s3.metric("Market high",   f"₹ {similar['price'].max():.2f} Cr")

                    diff = price - similar['price'].median()
                    pct  = diff / similar['price'].median() * 100
                    if abs(pct) < 5:
                        st.success("This estimate is at the market median.")
                    elif diff > 0:
                        st.info(f"This estimate is **₹{diff:.2f} Cr "
                                f"({pct:+.0f}%) above** the median.")
                    else:
                        st.info(f"This estimate is **₹{abs(diff):.2f} Cr "
                                f"({pct:+.0f}%) below** the median.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ─────────────────────────────────────────────────────────────
# PAGE 2 — RECOMMENDER
# ─────────────────────────────────────────────────────────────

elif page == "🔍 Recommender":
    st.markdown('<div class="page-title">Find similar properties</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-caption">Set your preferences — the '
                'recommender finds the closest matching properties from the '
                'database.</div>', unsafe_allow_html=True)

    rec = get_recommender()

    col_a, col_b, col_c, col_d = st.columns([1.2, 1.6, 1, 1])
    with col_a:
        rec_city = st.selectbox("City", rec.cities(), key="rec_city")
    with col_b:
        rec_sector_options = ['Any'] + get_sectors_for_city(rec_city)
        rec_sector  = st.selectbox(
            f"Location ({len(rec_sector_options)-1} options)",
            rec_sector_options,
            key=f"rec_sector_{rec_city}",
        )
    with col_c:
        rec_type    = st.selectbox("Type", ["Any", "flat", "house"],
                                   key="rec_type")
    with col_d:
        rec_bhk     = st.selectbox("BHK", ["Any", 1, 2, 3, 4, 5],
                                   key="rec_bhk")

    col_e, col_f = st.columns([3, 1])
    with col_e:
        rec_budget  = st.slider("Max budget (₹ Cr)",
                                0.5, 50.0, 10.0, step=0.5, key="rec_budget")
    with col_f:
        rec_n       = st.number_input("Show top", 3, 20, 6, key="rec_n")

    with st.expander("Adjust similarity weights"):
        st.caption("Higher weight = that dimension matters more.")
        w1, w2, w3 = st.columns(3)
        with w1:
            w_num = st.slider("Numerical (price/area/rooms)",
                              0, 50, 30, key="w_num")
        with w2:
            w_cat = st.slider("Categorical (type/luxury/furnishing)",
                              0, 50, 20, key="w_cat")
        with w3:
            w_loc = st.slider("Location (sector/city)",
                              0, 50, 8, key="w_loc")

    if st.button("Find similar", type="primary", use_container_width=True):
        rec.update_weights(w_num, w_cat, w_loc)
        results = rec.recommend_by_filters(
            city          = rec_city,
            sector        = rec_sector if rec_sector != 'Any' else None,
            property_type = rec_type   if rec_type   != 'Any' else None,
            bedrooms      = int(rec_bhk) if rec_bhk != 'Any' else None,
            budget_max    = rec_budget,
            top_n         = rec_n,
        )

        if results.empty:
            st.warning("No matches — try relaxing the filters or "
                       "increasing the budget.")
        else:
            st.write("")
            top_score = results['SimilarityScore'].max()
            avg_score = results['SimilarityScore'].mean()

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Matches",   len(results))
            mc2.metric("Top score", f"{top_score:.2f}")
            mc3.metric("Avg score", f"{avg_score:.2f}")
            st.write("")

            cols = st.columns(3)
            for i, row in results.iterrows():
                col = cols[i % 3]
                norm  = row['SimilarityScore'] / top_score
                color = ("#1D9E75" if norm > 0.85 else
                         "#378ADD" if norm > 0.65 else
                         "#EF9F27")
                psqft = int(row['price'] * 10_000_000 / row['built_up_area'])

                with col:
                    st.markdown(f"""
                    <div class="property-card" style="border-top: 4px solid {color};">
                        <div class="rank">#{int(row['Rank'])} · match {row['SimilarityScore']:.0f}</div>
                        <div class="name">{row['property_type'].title()} in {row['sector']}</div>
                        <div class="meta">
                            ₹ <b>{row['price']:.2f} Cr</b> · {int(row['bedRoom'])} BHK ·
                            {int(row['built_up_area'])} sqft<br>
                            {row['agePossession']} · {row['luxury_category']} luxury<br>
                            <span style="opacity:0.7;">₹ {psqft:,}/sqft</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with st.expander("View full results table"):
                cols_show = ['Rank', 'sector', 'property_type', 'bedRoom',
                             'price', 'built_up_area', 'agePossession',
                             'luxury_category', 'SimilarityScore']
                st.dataframe(
                    results[cols_show],
                    use_container_width=True,
                    hide_index=True,
                )


# ─────────────────────────────────────────────────────────────
# PAGE 3 — INSIGHTS  (rich, multi-section)
# ─────────────────────────────────────────────────────────────

elif page == "📊 Insights":
    st.markdown('<div class="page-title">Market insights</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-caption">Maps, distributions, drivers and '
                'comparisons across Gurgaon and Bangalore.</div>',
                unsafe_allow_html=True)

    ins = get_insights()

    # ── KPI HEADER (city comparison at a glance)
    st.markdown("##### Both cities at a glance")
    comp = ins.city_comparison()
    st.dataframe(comp, use_container_width=True, hide_index=True)
    st.divider()

    # ── CITY SELECTOR
    insight_city = st.selectbox("City to analyse",
                                ins.cities() + ['both'], key="ins_city")
    if insight_city == 'both':
        sub = df_master.copy()
        title_city = "Gurgaon + Bangalore"
    else:
        sub = df_master[df_master['city'] == insight_city]
        title_city = insight_city.title()

    cv = ins.cv_score(insight_city if insight_city != 'both' else None)
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Properties",   len(sub))
    mc2.metric("Median price", f"₹ {sub['price'].median():.2f} Cr")
    mc3.metric("Locations",    sub['sector'].nunique())
    mc4.metric("Model R²",     f"{cv['r2_mean']:.2f}",
               f"±{cv['r2_std']:.2f}")

    st.write("")

    # ── 6 TABS for deep analysis
    insight_tabs = st.tabs([
        "🗺 Map",
        "📊 Distributions",
        "🏘 Locations",
        "💰 Price drivers",
        "🆚 City comparison",
        "🔬 Custom impact",
    ])

    # ═══════════════════════════════════════════════════════════
    # TAB 1 — MAP
    # ═══════════════════════════════════════════════════════════
    with insight_tabs[0]:
        st.markdown(f"##### Property locations in {title_city}")

        map_col1, map_col2 = st.columns([3, 1])
        with map_col2:
            map_metric = st.radio(
                "Color by",
                ["Price", "Property count", "Price per sqft"],
                key="map_metric",
            )
            map_max_points = st.slider("Max markers",
                                       100, 4000, 1500, 100,
                                       key="map_max")

        # Aggregate by sector for cleaner map
        agg = sub.groupby('sector').agg(
            n        = ('price', 'count'),
            price    = ('price', 'median'),
            area     = ('built_up_area', 'median'),
        ).reset_index()
        agg['price_per_sqft'] = (agg['price'] * 10_000_000 / agg['area']).round(0)

        # Add city back so geo lookup uses it
        agg['city'] = agg['sector'].apply(
            lambda s: 'bangalore' if 'bangalore' in str(s).lower()
                      or s in df_master[df_master['city']=='bangalore']['sector'].values
                      else 'gurgaon')
        agg = add_coords(agg, sector_col='sector', city_col='city')
        agg = agg.dropna(subset=['lat', 'lon']).head(map_max_points)

        with map_col1:
            if agg.empty:
                st.warning("No mappable locations found.")
            else:
                # Colour map: scale chosen metric to color
                if map_metric == "Price":
                    metric_col = 'price'
                    label = '₹ Cr'
                elif map_metric == "Property count":
                    metric_col = 'n'
                    label = 'count'
                else:
                    metric_col = 'price_per_sqft'
                    label = '₹/sqft'

                # Normalise to size for st.map (size in metres)
                v = agg[metric_col]
                vmin, vmax = float(v.min()), float(v.max())
                norm = (v - vmin) / (vmax - vmin + 1e-9)
                agg['size'] = (norm * 250 + 50).astype(int)

                # Color gradient (red = high, blue = low)
                def _color(n):
                    r = int(255 * n)
                    b = int(255 * (1 - n))
                    return [r, 80, b, 180]
                agg['color'] = norm.apply(_color)

                # Use pydeck for size + color
                try:
                    import pydeck as pdk
                    layer = pdk.Layer(
                        'ScatterplotLayer',
                        data=agg.rename(columns={'lat': 'latitude',
                                                 'lon': 'longitude'}),
                        get_position=['longitude', 'latitude'],
                        get_radius='size',
                        get_fill_color='color',
                        pickable=True,
                        opacity=0.7,
                    )
                    centre_lat = float(agg['lat'].median())
                    centre_lon = float(agg['lon'].median())
                    deck = pdk.Deck(
                        layers=[layer],
                        initial_view_state=pdk.ViewState(
                            latitude=centre_lat,
                            longitude=centre_lon,
                            zoom=10,
                            pitch=0,
                        ),
                        tooltip={
                            "html": "<b>{sector}</b><br>"
                                    "Properties: {n}<br>"
                                    "Median price: ₹ {price} Cr<br>"
                                    "Price/sqft: ₹ {price_per_sqft}",
                        },
                        map_style=None,
                    )
                    st.pydeck_chart(deck, use_container_width=True)
                except Exception:
                    # Fallback: plain st.map
                    st.map(agg[['lat', 'lon']], size=agg['size'].tolist())

        st.caption(f"Showing {len(agg)} sectors. Bigger / redder dot = higher "
                   f"{map_metric.lower()}.")

        with st.expander("View location data table"):
            show = agg[['sector', 'n', 'price',
                        'price_per_sqft', 'area', 'lat', 'lon']]
            show.columns = ['Sector', 'Listings', 'Median ₹Cr',
                            '₹/sqft', 'Median sqft', 'Lat', 'Lng']
            st.dataframe(show.sort_values('Median ₹Cr', ascending=False),
                         use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════════
    # TAB 2 — DISTRIBUTIONS
    # ═══════════════════════════════════════════════════════════
    with insight_tabs[1]:
        st.markdown(f"##### How {title_city} properties are distributed")

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Price distribution (₹ Cr)**")
            price_dist = (
                pd.cut(sub['price'], bins=20)
                  .value_counts().sort_index()
                  .rename("count")
            )
            price_dist.index = price_dist.index.astype(str)
            st.bar_chart(price_dist, height=300)

        with d2:
            st.markdown("**Built-up area distribution (sqft)**")
            area_dist = (
                pd.cut(sub['built_up_area'], bins=20)
                  .value_counts().sort_index()
                  .rename("count")
            )
            area_dist.index = area_dist.index.astype(str)
            st.bar_chart(area_dist, height=300)

        st.write("")
        d3, d4 = st.columns(2)
        with d3:
            st.markdown("**Bedrooms breakdown**")
            bhk_dist = sub['bedRoom'].value_counts().sort_index()
            bhk_dist.index = bhk_dist.index.astype(int).astype(str) + " BHK"
            st.bar_chart(bhk_dist, height=280)

        with d4:
            st.markdown("**Property type split**")
            type_dist = sub['property_type'].value_counts()
            st.bar_chart(type_dist, height=280)

        st.write("")
        d5, d6 = st.columns(2)
        with d5:
            st.markdown("**Furnishing**")
            furn_map = {0: 'Unfurnished', 1: 'Semi-furnished', 2: 'Furnished',
                        0.0: 'Unfurnished', 1.0: 'Semi-furnished', 2.0: 'Furnished'}
            furn = sub['furnishing_type'].map(furn_map).fillna(
                sub['furnishing_type'].astype(str)).value_counts()
            st.bar_chart(furn, height=280)

        with d6:
            st.markdown("**Luxury level**")
            lux_order = ['Low', 'Medium', 'High']
            lux = sub['luxury_category'].value_counts().reindex(
                lux_order, fill_value=0)
            st.bar_chart(lux, height=280)

        st.write("")
        st.markdown("##### Price vs Built-up area (relationship)")
        scatter_data = sub[['built_up_area', 'price', 'property_type']].copy()
        scatter_data.columns = ['Built-up area (sqft)', 'Price (₹ Cr)',
                                'Type']
        # Sample to keep chart fast
        if len(scatter_data) > 1500:
            scatter_data = scatter_data.sample(1500, random_state=42)
        st.scatter_chart(
            scatter_data,
            x='Built-up area (sqft)',
            y='Price (₹ Cr)',
            color='Type',
            height=400,
        )

    # ═══════════════════════════════════════════════════════════
    # TAB 3 — LOCATIONS (sector-level analysis)
    # ═══════════════════════════════════════════════════════════
    with insight_tabs[2]:
        st.markdown(f"##### Sector-level analysis for {title_city}")

        loc_metric = st.radio(
            "Rank sectors by",
            ["Median price (₹ Cr)", "Price per sqft (₹)",
             "Number of listings", "Median area (sqft)"],
            horizontal=True, key="loc_metric",
        )
        loc_n = st.slider("Show top", 5, 30, 12, 1, key="loc_n")

        loc_stats = sub.groupby('sector').agg(
            listings=('price', 'count'),
            median_price=('price', 'median'),
            min_price=('price', 'min'),
            max_price=('price', 'max'),
            median_area=('built_up_area', 'median'),
        ).reset_index()
        loc_stats['price_per_sqft'] = (
            loc_stats['median_price'] * 10_000_000 / loc_stats['median_area']
        ).round(0).astype(int)

        # Need at least 3 listings to be representative
        loc_stats = loc_stats[loc_stats['listings'] >= 3]

        sort_map = {
            "Median price (₹ Cr)":   ('median_price', False),
            "Price per sqft (₹)":    ('price_per_sqft', False),
            "Number of listings":    ('listings', False),
            "Median area (sqft)":    ('median_area', False),
        }
        sort_col, asc = sort_map[loc_metric]
        ranked = loc_stats.sort_values(sort_col, ascending=asc).head(loc_n)

        l1, l2 = st.columns([2, 1])
        with l1:
            chart_df = ranked.set_index('sector')[[sort_col]]
            st.bar_chart(chart_df, height=400, horizontal=True)
        with l2:
            st.markdown(f"**Top {loc_n}**")
            display = ranked.copy()
            display['median_price'] = display['median_price'].round(2)
            display['median_area']  = display['median_area'].astype(int)
            display = display.rename(columns={
                'sector': 'Sector',
                'listings': '#',
                'median_price': '₹ Cr',
                'price_per_sqft': '₹/sqft',
            })
            st.dataframe(
                display[['Sector', '#', '₹ Cr', '₹/sqft']],
                use_container_width=True, hide_index=True,
            )

        st.write("")
        st.markdown("##### Most affordable vs most expensive locations")
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("**Cheapest 5**")
            cheap = (loc_stats.sort_values('median_price').head(5)
                     [['sector', 'listings', 'median_price']]
                     .rename(columns={'sector': 'Sector',
                                      'listings': '#',
                                      'median_price': '₹ Cr'}))
            cheap['₹ Cr'] = cheap['₹ Cr'].round(2)
            st.dataframe(cheap, use_container_width=True, hide_index=True)
        with a2:
            st.markdown("**Priciest 5**")
            pricey = (loc_stats.sort_values('median_price', ascending=False).head(5)
                      [['sector', 'listings', 'median_price']]
                      .rename(columns={'sector': 'Sector',
                                       'listings': '#',
                                       'median_price': '₹ Cr'}))
            pricey['₹ Cr'] = pricey['₹ Cr'].round(2)
            st.dataframe(pricey, use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════════
    # TAB 4 — PRICE DRIVERS (Ridge regression)
    # ═══════════════════════════════════════════════════════════
    with insight_tabs[3]:
        st.markdown(f"##### What pushes {title_city} prices up or down")
        st.caption("Ridge regression coefficients — log-price change per "
                   "standardised unit of each feature. Positive = higher price.")

        target_city = insight_city if insight_city != 'both' else None
        coef = ins.feature_importance(target_city, top_n=20)
        coef['readable'] = coef['feature'].apply(_humanize_feature)

        # Two-column layout: chart + numeric features only
        chart_df = coef.set_index('readable')[['coefficient']].sort_values(
            'coefficient')
        st.bar_chart(chart_df, height=450)

        st.write("")
        # Highlight numeric (non-OHE) features separately
        num_only = coef[~coef['feature'].str.startswith(
            ('sector_', 'agePossession_'))].head(10)
        if not num_only.empty:
            st.markdown("**Core numeric / categorical features (excluding sectors)**")
            num_chart = num_only.set_index('readable')[['coefficient']].sort_values(
                'coefficient')
            st.bar_chart(num_chart, height=300)

        with st.expander("View full coefficient table"):
            display = coef[['readable', 'coefficient', 'direction']]
            display.columns = ['Feature', 'Coefficient', 'Effect']
            st.dataframe(display, use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════════
    # TAB 5 — CITY COMPARISON
    # ═══════════════════════════════════════════════════════════
    with insight_tabs[4]:
        st.markdown("##### Direct comparison: Gurgaon vs Bangalore")

        # Side-by-side stats
        comp_data = []
        for c in df_master['city'].unique():
            s = df_master[df_master['city'] == c]
            comp_data.append({
                'city': c.title(),
                'Total': len(s),
                'Flats': int((s['property_type'] == 'flat').sum()),
                'Houses': int((s['property_type'] == 'house').sum()),
                'Median ₹ Cr': round(s['price'].median(), 2),
                'Mean ₹ Cr': round(s['price'].mean(), 2),
                'Median sqft': int(s['built_up_area'].median()),
                'Median ₹/sqft': int(
                    s['price'].median() * 10_000_000 / s['built_up_area'].median()),
                'Locations': s['sector'].nunique(),
            })
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.write("")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Median price by BHK**")
            bhk_compare = (df_master.groupby(['bedRoom', 'city'])['price']
                           .median().unstack(fill_value=0).round(2))
            bhk_compare.index = bhk_compare.index.astype(int).astype(str) + " BHK"
            st.bar_chart(bhk_compare, height=320)

        with cc2:
            st.markdown("**Median price by property type**")
            type_compare = (df_master.groupby(['property_type', 'city'])['price']
                            .median().unstack(fill_value=0).round(2))
            st.bar_chart(type_compare, height=320)

        st.write("")
        cc3, cc4 = st.columns(2)
        with cc3:
            st.markdown("**Median price by luxury level**")
            lux_compare = (df_master.groupby(['luxury_category', 'city'])['price']
                           .median().unstack(fill_value=0).round(2))
            lux_compare = lux_compare.reindex(['Low', 'Medium', 'High'])
            st.bar_chart(lux_compare, height=320)

        with cc4:
            st.markdown("**Listings by age category**")
            age_compare = (df_master.groupby(['agePossession', 'city']).size()
                           .unstack(fill_value=0))
            st.bar_chart(age_compare, height=320)

        st.write("")
        st.markdown("##### Affordability heatmap — median ₹ Cr by BHK and city")
        heat = (df_master.groupby(['bedRoom', 'city'])['price']
                .median().unstack(fill_value=0).round(2))
        heat.index = heat.index.astype(int).astype(str) + " BHK"
        st.dataframe(heat.style.background_gradient(cmap='RdYlGn_r', axis=None),
                     use_container_width=True)

    # ═══════════════════════════════════════════════════════════
    # TAB 6 — CUSTOM IMPACT
    # ═══════════════════════════════════════════════════════════
    with insight_tabs[5]:
        st.markdown("##### How much does each factor change the price?")
        st.caption(f"Based on the {title_city} Ridge model.")

        target_city = insight_city if insight_city != 'both' else None
        feat_choice = st.selectbox(
            "Which factor?",
            ["built_up_area", "bedRoom", "bathroom",
             "property_type", "luxury_category", "furnishing_type"],
            key="ins_feat",
        )
        delta_label = ("sqft" if feat_choice == "built_up_area"
                       else "level/unit")
        delta_val = st.number_input(
            f"Change by ({delta_label})",
            value=100.0 if feat_choice == "built_up_area" else 1.0,
            step=10.0  if feat_choice == "built_up_area" else 1.0,
        )

        try:
            impact = ins.predict_impact(target_city or 'gurgaon',
                                        feat_choice, delta_val)
            ic1, ic2 = st.columns(2)
            ic1.metric("Estimated price change",
                       f"{impact['price_pct']:+.2f}%")
            median_price = sub['price'].median()
            cr_change = median_price * impact['price_pct'] / 100
            ic2.metric("Approx. ₹ change (median property)",
                       f"₹ {cr_change:+.2f} Cr")

            st.caption(
                f"Increasing **{feat_choice}** by **{delta_val} {delta_label}** "
                f"is associated with a **{impact['price_pct']:+.2f}%** change "
                f"in price (model coefficient: {impact['coefficient']:+.4f})."
            )
        except Exception as e:
            st.error(f"Impact calculation failed: {e}")
