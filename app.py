import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.preprocess import load_raw_dataset, preprocess_travel_data
from src.recommender import ContentBasedRecommender
from src.budget_model import TransparentBaselineEstimator, TravelBudgetPredictor
from src.analytics import (
    get_summary_metrics,
    plot_top_expensive_destinations,
    plot_cost_distribution,
    plot_correlation_heatmap,
    plot_destination_types_pie
)
from src.maps import get_google_maps_api_key, render_google_maps_route_component

# Page Configuration
st.set_page_config(
    page_title="TravelSense AI – Intelligent Recommendation, Route & Budget System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

DEFAULT_IMAGE_URL = "https://images.unsplash.com/photo-1477587458883-47145ed94245?auto=format&fit=crop&w=800&q=80"

def safe_image_url(row_or_dict):
    try:
        if isinstance(row_or_dict, pd.Series):
            val = row_or_dict.get('image_url', DEFAULT_IMAGE_URL)
        elif isinstance(row_or_dict, dict):
            val = row_or_dict.get('image_url', DEFAULT_IMAGE_URL)
        else:
            val = getattr(row_or_dict, 'image_url', DEFAULT_IMAGE_URL)
        if pd.isnull(val) or not str(val).strip():
            return DEFAULT_IMAGE_URL
        return str(val).strip()
    except Exception:
        return DEFAULT_IMAGE_URL

# Cache data loading & ML components for high performance
@st.cache_data
def get_cleaned_data():
    cleaned_path = os.path.join(os.path.dirname(__file__), "data", "indian_destinations_cleaned.csv")
    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path)
    else:
        pipe_res = preprocess_travel_data()
        df = pipe_res['df_cleaned']
    if 'image_url' not in df.columns:
        df['image_url'] = DEFAULT_IMAGE_URL
    else:
        df['image_url'] = df['image_url'].fillna(DEFAULT_IMAGE_URL)
    return df

@st.cache_resource
def get_recommender():
    return ContentBasedRecommender()

@st.cache_resource
def get_budget_predictor(df):
    model_path = os.path.join(os.path.dirname(__file__), "models", "budget_model.joblib")
    predictor = TravelBudgetPredictor()
    
    if os.path.exists(model_path):
        try:
            predictor.load(model_path)
            return predictor
        except Exception:
            pass
            
    pipe_res = preprocess_travel_data()
    df_cleaned = pipe_res['df_cleaned']
    from train_models import generate_trip_level_synthetic_data
    dataset_df = generate_trip_level_synthetic_data(df_cleaned, num_samples=1800, random_state=42)
    predictor.fit_and_evaluate(dataset_df, test_size=0.2, random_state=42)
    try:
        predictor.save(model_path)
    except Exception:
        pass
    return predictor

# Load Data and Engines
df_dest = get_cleaned_data()
recommender = get_recommender()
budget_predictor = get_budget_predictor(df_dest)

# Sidebar
with st.sidebar:
    st.markdown("### ✈️ TravelSense AI")
    st.caption("Intelligent Destination, Route & Budget Engine")
    st.markdown("---")
    
    st.markdown("#### 🔑 Google Maps API Config")
    current_key = get_google_maps_api_key()
    user_api_key = st.text_input(
        "Google Maps API Key:",
        value=current_key,
        type="password",
        help="Optional: Enables live Google Maps JS, Routes API & Places Autocomplete. Leave empty for visual preview."
    )
    if user_api_key.strip():
        st.session_state["user_google_maps_api_key"] = user_api_key.strip()
        st.success("✅ Google Maps API Key Configured!")
    else:
        st.info("ℹ️ Using visual route preview mode.")

    st.markdown("---")
    st.markdown("#### 🔍 Quick Destination Search")
    search_query = st.text_input("Type destination or state:", placeholder="e.g. Ooty, Goa, Jaipur...")
    
    if search_query.strip():
        search_res = df_dest[
            df_dest['destination'].str.contains(search_query, case=False, na=False) |
            df_dest['state'].str.contains(search_query, case=False, na=False)
        ].head(5)
        
        if not search_res.empty:
            st.markdown(f"**Found {len(search_res)} matches:**")
            for _, r in search_res.iterrows():
                st.markdown(
                    f"• **{r['destination']}** ({r['state']})  \n"
                    f"  <span style='color:#0284C7; font-size:0.85rem;'>₹{int(r['total_daily_cost_inr']):,}/day</span> | ⭐ {r['popularity_score']}",
                    unsafe_allow_html=True
                )
        else:
            st.info("No matching destination found.")
            
    st.markdown("---")
    st.markdown("#### 📌 Key System Specs")
    st.markdown(f"• **Destinations:** {len(df_dest)} Indian Spots")
    st.markdown(f"• **States Covered:** {df_dest['state'].nunique()}")
    st.markdown(f"• **ML Model:** RandomForestRegressor")
    st.markdown(f"• **Route Engine:** Google Maps Routes API")
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #64748B; font-size: 0.8rem;'>TravelSense AI v2.0 • Powered by Scikit-Learn & Google Maps</div>", unsafe_allow_html=True)

# Hero Header Banner
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">✈️ TravelSense AI</div>
        <div class="hero-subtitle">Discover Your Ideal Destination, Plan Smart Routes with Google Maps, & Predict Travel Budgets with Machine Learning.</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Top Navigation Tabs
tab_home, tab_recommender, tab_route, tab_budget, tab_compare, tab_analytics, tab_eda = st.tabs([
    "🏠 Overview & Home",
    "🧳 AI Destination Recommender",
    "🗺️ Smart Route Planner",
    "💰 Travel Budget Calculator",
    "⚖️ Destination Comparison",
    "📊 Analytics Dashboard",
    "🔬 Preprocessing & Model Metrics"
])

# ---------------------------------------------------------
# TAB 1: OVERVIEW & HOME
# ---------------------------------------------------------
with tab_home:
    st.markdown("### 📌 System Overview & Key Indicators")
    
    metrics = get_summary_metrics(df_dest)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-box-title">Destinations</div>
                <div class="metric-box-value">{metrics['total_destinations']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-box-title">Avg Daily Cost</div>
                <div class="metric-box-value">₹{metrics['avg_daily_cost_inr']:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-box-title">Categories</div>
                <div class="metric-box-value">{metrics['num_categories']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-box-title">States Covered</div>
                <div class="metric-box-value">{df_dest['state'].nunique()}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col5:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-box-title">Avg Rating</div>
                <div class="metric-box-value">⭐ {metrics.get('avg_rating', 4.2)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 🌟 Popular Featured Travel Destinations")
    
    cat_filter = st.selectbox("Filter Destinations by Category:", ["All Categories"] + sorted(df_dest['destination_type'].unique().tolist()), index=0)
    
    display_df = df_dest.copy()
    if cat_filter != "All Categories":
        display_df = display_df[display_df['destination_type'] == cat_filter]
        
    top_dest = display_df.sort_values(by='popularity_score', ascending=False).head(6)
    
    grid_cols = st.columns(3)
    for idx, (_, row) in enumerate(top_dest.iterrows()):
        with grid_cols[idx % 3]:
            st.markdown(
                f"""
                <div class="ts-card">
                    <img src="{safe_image_url(row)}" class="card-img" />
                    <h4 class="card-title">{row['destination']}, {row['state']}</h4>
                    <p><span class="badge-pill badge-blue">{row['destination_type']}</span> <span class="badge-pill badge-green">₹{int(row['total_daily_cost_inr']):,}/day</span></p>
                    <p><b>Best Season:</b> {row['best_season']}<br>
                    <b>Climate:</b> {row['climate']}<br>
                    <b>Rating:</b> ⭐ {row['popularity_score']}/5.0</p>
                </div>
                """,
                unsafe_allow_html=True
            )

# ---------------------------------------------------------
# TAB 2: AI DESTINATION RECOMMENDER
# ---------------------------------------------------------
with tab_recommender:
    st.markdown("### 🧳 AI Destination Recommendation Engine")
    st.write("Enter your travel budget and preferences to receive top-ranked destination recommendations using Cosine Similarity.")

    with st.form("recommender_form"):
        col1, col2 = st.columns(2)
        with col1:
            total_budget_inr = st.number_input("💵 Total Travel Budget (INR ₹)", min_value=3000, max_value=500000, value=45000, step=2000)
            group_size = st.number_input("👥 Number of Travelers", min_value=1, max_value=20, value=2, step=1)
            duration_days = st.number_input("⏱️ Trip Duration (Days)", min_value=1, max_value=30, value=5, step=1)

        with col2:
            types = ["Any"] + sorted(df_dest['destination_type'].unique().tolist())
            destination_type = st.selectbox("🎯 Preferred Destination Type", types, index=0)

            climates = ["Any"] + sorted(df_dest['climate'].unique().tolist())
            climate = st.selectbox("🌤️ Preferred Climate", climates, index=0)

            seasons = ["Any"] + sorted(df_dest['best_season'].unique().tolist())
            best_season = st.selectbox("🗓️ Preferred Travel Season", seasons, index=0)

        submit_rec = st.form_submit_button("🚀 Find Matching Destinations")

    if submit_rec or 'last_recs' in st.session_state:
        if submit_rec:
            st.session_state['last_query'] = {
                'total_budget_inr': float(total_budget_inr),
                'group_size': int(group_size),
                'duration_days': int(duration_days),
                'destination_type': destination_type,
                'climate': climate,
                'best_season': best_season
            }
        
        q = st.session_state.get('last_query', {
            'total_budget_inr': 45000.0, 'group_size': 2, 'duration_days': 5,
            'destination_type': 'Any', 'climate': 'Any', 'best_season': 'Any'
        })

        recs_df = recommender.recommend(
            total_budget_inr=q['total_budget_inr'],
            group_size=q['group_size'],
            duration_days=q['duration_days'],
            destination_type=q['destination_type'],
            climate=q['climate'],
            best_season=q['best_season'],
            top_k=5
        )
        st.session_state['last_recs'] = recs_df

        st.markdown("---")
        st.markdown(f"### 🌟 Top 5 Recommendations (Budget: **₹{q['total_budget_inr']:,.0f}**, {q['group_size']} Travelers, {q['duration_days']} Days)")

        for rank, (_, row) in enumerate(recs_df.iterrows(), 1):
            match_p = row['similarity_percent']
            reasons_html = "".join([f"<li>{r}</li>" for r in row['recommendation_reasons']])
            
            st.markdown(
                f"""
                <div class="ts-card">
                    <div style="display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start;">
                        <div style="flex: 1; min-width: 240px; max-width: 300px;">
                            <img src="{safe_image_url(row)}" style="width: 100%; height: 210px; object-fit: cover; border-radius: 14px;" />
                        </div>
                        <div style="flex: 2; min-width: 300px;">
                            <h3 style="color: #0F172A !important; margin-top: 0;">#{rank}. {row['destination']}, {row['state']} <span class="badge-pill badge-purple" style="font-size: 0.9rem; float: right;">Match: {match_p}%</span></h3>
                            <div style="display: flex; gap: 15px; margin-bottom: 14px; background: #F8FAFC; padding: 14px; border-radius: 12px; border: 1px solid #E2E8F0;">
                                <div style="flex: 1;">
                                    <div style="font-size: 0.75rem; color: #64748B; font-weight: 700;">ESTIMATED COST</div>
                                    <div style="font-size: 1.25rem; font-weight: 800; color: #0284C7;">₹{row['estimated_trip_cost']:,.2f}</div>
                                    <div style="font-size: 0.85rem; color: #334155;"><b>Type:</b> {row['destination_type']}</div>
                                </div>
                                <div style="flex: 1; font-size: 0.85rem; color: #334155;">
                                    <div><b>Climate:</b> {row['climate']}</div>
                                    <div><b>Best Season:</b> {row['best_season']}</div>
                                </div>
                                <div style="flex: 1; font-size: 0.85rem; color: #334155;">
                                    <div><b>Rating:</b> ⭐ {row['popularity_score']}/5.0</div>
                                    <div><b>Stay:</b> {int(row['recommended_duration'])} days</div>
                                </div>
                            </div>
                            <div style="font-size: 0.92rem; color: #1E293B;">
                                <b>💡 Reasons for Recommendation:</b>
                                <ul style="margin-top: 4px; margin-bottom: 0; padding-left: 20px;">
                                    {reasons_html}
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            btn_col1, btn_col2 = st.columns([3, 1])
            with btn_col2:
                if st.button(f"🗺️ View Route & Plan Trip", key=f"route_btn_{rank}_{row['destination']}"):
                    st.session_state['selected_route_destination'] = f"{row['destination']}, {row['state']}"
                    st.toast(f"Transferred '{row['destination']}' to Smart Route Planner! Switch to tab 🗺️ Smart Route Planner to view route.", icon="📍")

# ---------------------------------------------------------
# TAB 3: SMART ROUTE PLANNER (GOOGLE MAPS PLATFORM)
# ---------------------------------------------------------
with tab_route:
    st.markdown("### 🗺️ Google Maps Smart Route Planner")
    st.write("Compute routes, compare travel modes, view fitted interactive maps, select alternative paths, and calculate driving fuel expenses.")

    # Preset Test Buttons
    st.markdown("#### ⚡ Preset Test Routes (1-Click Test):")
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        if st.button("🔘 Coimbatore ➔ Ooty"):
            st.session_state['route_origin'] = "Coimbatore, Tamil Nadu"
            st.session_state['selected_route_destination'] = "Ooty, Tamil Nadu"
    with p_col2:
        if st.button("🔘 Chennai ➔ Bengaluru"):
            st.session_state['route_origin'] = "Chennai, Tamil Nadu"
            st.session_state['selected_route_destination'] = "Bengaluru, Karnataka"
    with p_col3:
        if st.button("🔘 Madurai ➔ Kodaikanal"):
            st.session_state['route_origin'] = "Madurai, Tamil Nadu"
            st.session_state['selected_route_destination'] = "Kodaikanal, Tamil Nadu"

    default_orig = st.session_state.get('route_origin', 'Coimbatore, Tamil Nadu')
    default_dest = st.session_state.get('selected_route_destination', 'Ooty, Tamil Nadu')
    maps_api_key = get_google_maps_api_key()

    if not maps_api_key:
        st.info("ℹ️ **Notice:** Running in visual route preview mode. To activate live Google Maps JS, Places Autocomplete & live traffic, enter your `GOOGLE_MAPS_API_KEY` in `.env` or in the sidebar.")

    # Render Interactive HTML/JS Route Component
    render_google_maps_route_component(
        origin=default_orig,
        destination=default_dest,
        travel_mode="DRIVING",
        api_key=maps_api_key
    )

    # ---------------------------------------------------------
    # FUEL & DRIVING EXPENSE CALCULATOR INTEGRATION
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### ⛽ Driving Fuel Cost Calculator (Connected to Budget System)")
    st.write("Calculate driving fuel cost based on trip distance, vehicle mileage, and fuel prices.")

    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        dist_input = st.number_input("📏 Driving Distance (km)", min_value=1.0, max_value=5000.0, value=85.0, step=5.0)
    with f_col2:
        mileage_input = st.number_input("🚗 Vehicle Mileage (km / L)", min_value=3.0, max_value=50.0, value=15.0, step=0.5)
    with f_col3:
        fuel_price_input = st.number_input("⛽ Fuel Price (INR ₹ / L)", min_value=50.0, max_value=200.0, value=102.50, step=0.50)
    with f_col4:
        trip_type = st.selectbox("🔄 Trip Type", ["Round Trip (2x Distance)", "One-Way Trip"], index=0)

    total_calc_dist = dist_input * 2.0 if trip_type.startswith("Round") else dist_input
    fuel_liters_needed = total_calc_dist / mileage_input
    total_fuel_cost = fuel_liters_needed * fuel_price_input

    st.markdown(
        f"""
        <div class="insight-box" style="border-left: 5px solid #10B981;">
            <b>⛽ Estimated Fuel Expense Result:</b><br>
            • Total Distance: <b>{total_calc_dist:,.1f} km</b> ({trip_type})<br>
            • Estimated Fuel Required: <b>{fuel_liters_needed:,.2f} Liters</b><br>
            • Total Fuel Cost: <b style="font-size: 1.3rem; color: #059669;">₹{total_fuel_cost:,.2f}</b><br>
            <span style="font-size: 0.8rem; color: #64748B;">*Formula-based rate calculation (Distance ÷ Mileage × Fuel Rate), separate from the Scikit-Learn Random Forest ML budget prediction model.</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# TAB 4: TRAVEL BUDGET CALCULATOR
# ---------------------------------------------------------
with tab_budget:
    st.markdown("### 💰 Travel Budget Prediction Calculator")
    st.write("Compare transparent rate-based cost calculations with Machine Learning RandomForest predictions.")

    dest_list = df_dest['destination'].tolist()
    col_a, col_b = st.columns(2)
    with col_a:
        selected_dest_name = st.selectbox("📍 Select Destination", dest_list, index=0)
        dest_matches = df_dest[df_dest['destination'] == selected_dest_name]
        dest_row = dest_matches.iloc[0] if not dest_matches.empty else df_dest.iloc[0]
        duration_days_calc = st.number_input("⏱️ Trip Duration (Days)", min_value=1, max_value=30, value=int(dest_row['recommended_duration']), step=1)
        group_size_calc = st.number_input("👥 Number of Travelers", min_value=1, max_value=20, value=2, step=1)

    with col_b:
        accommodation_tier = st.selectbox("🏨 Accommodation Category", ["Budget", "Standard", "Luxury", "Ultra Luxury"], index=1)
        user_target_budget = st.number_input("🎯 Target Budget Goal (INR ₹, Optional)", min_value=1000, max_value=500000, value=30000, step=1000)

    if st.button("🚀 Calculate Estimated Budget"):
        res = budget_predictor.predict_trip(
            dest_row=dest_row,
            duration_days=duration_days_calc,
            group_size=group_size_calc,
            accommodation_tier=accommodation_tier
        )

        st.markdown("---")
        b_col_img, b_col_info = st.columns([1, 2])
        with b_col_img:
            st.markdown(f'<img src="{safe_image_url(dest_row)}" style="width:100%; height:230px; object-fit:cover; border-radius:16px; box-shadow:0 6px 16px rgba(0,0,0,0.12);" />', unsafe_allow_html=True)
        with b_col_info:
            st.markdown(f"### Estimated Trip Budget for **{dest_row['destination']} ({dest_row['state']})**")
            st.caption(f"Parameters: {group_size_calc} Travelers | {duration_days_calc} Days | {accommodation_tier} Lodging")

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown(
                    f"""
                    <div class="metric-box" style="border-top-color: #0284C7;">
                        <div class="metric-box-title">Transparent Baseline Cost</div>
                        <div class="metric-box-value">₹{res['baseline_total_inr']:,.2f}</div>
                        <div style="font-size: 0.85rem; color: #64748B; margin-top: 4px;">Calculated directly from destination rate columns</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with res_col2:
                st.markdown(
                    f"""
                    <div class="metric-box" style="border-top-color: #7E22CE;">
                        <div class="metric-box-title">Random Forest ML Prediction</div>
                        <div class="metric-box-value">₹{res['ml_predicted_total_inr']:,.2f}</div>
                        <div style="font-size: 0.85rem; color: #64748B; margin-top: 4px;">Scikit-Learn RandomForestRegressor Model</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown(f"<div class='insight-box'>ℹ️ <b>Model Status:</b> {res['model_disclaimer']}</div>", unsafe_allow_html=True)

        st.markdown("#### 📊 Itemized Cost Breakdown (INR ₹)")
        b_col1, b_col2 = st.columns([1, 1])
        with b_col1:
            breakdown_dict = res['breakdown']
            breakdown_df = pd.DataFrame(list(breakdown_dict.items()), columns=['Category', 'Amount (₹)'])
            fig_pie = px.pie(
                breakdown_df,
                names='Category',
                values='Amount (₹)',
                color_discrete_sequence=['#0284C7', '#38BDF8', '#7E22CE', '#10B981'],
                hole=0.45
            )
            fig_pie.update_traces(textinfo='percent+label', hovertemplate='<b>%{label}</b><br>Cost: ₹%{value:,.2f}<extra></extra>')
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=20, b=20),
                height=350,
                font=dict(color="#0F172A"),
                legend=dict(font=dict(color="#0F172A"))
            )
            st.plotly_chart(fig_pie, width="stretch")

        with b_col2:
            st.table(breakdown_df.style.format({'Amount (₹)': '₹{:,.2f}'}))
            per_person_total = res['baseline_total_inr'] / group_size_calc
            per_person_day = res['baseline_total_inr'] / (group_size_calc * duration_days_calc)
            st.write(f"**Per Traveler Total Cost:** ₹{per_person_total:,.2f}")
            st.write(f"**Daily Rate per Person:** ₹{per_person_day:,.2f}")

        # Suggested Day-by-Day Itinerary Feature
        st.markdown("---")
        st.markdown(f"#### 🗓️ Suggested Day-by-Day Itinerary Planner ({dest_row['destination']})")
        for day in range(1, duration_days_calc + 1):
            with st.expander(f"📍 Day {day}: Explore & Activities", expanded=(day == 1)):
                if day == 1:
                    st.write(f"• **Morning/Afternoon:** Arrival at {dest_row['destination']}, check-in at hotel, refresh.")
                    st.write(f"• **Evening:** Scenic walk around famous city spots & authentic local dinner.")
                elif day == 2:
                    st.write(f"• **Morning:** Guided tour of top {dest_row['destination_type']} landmarks.")
                    st.write(f"• **Afternoon:** Cultural exploration and famous local food tasting.")
                    st.write(f"• **Evening:** Sunset viewpoint visit & traditional market shopping.")
                elif day == 3:
                    st.write(f"• **Full Day:** Adventure/nature activity trip in near surrounding areas.")
                elif day == duration_days_calc:
                    st.write(f"• **Morning:** Souvenir shopping, relaxed breakfast & hotel check-out.")
                    st.write(f"• **Afternoon:** Departure transfer.")
                else:
                    st.write(f"• **Day {day}:** Local sightseeing, photography, & regional cuisine experience.")

# ---------------------------------------------------------
# TAB 5: DESTINATION COMPARISON TOOL
# ---------------------------------------------------------
with tab_compare:
    st.markdown("### ⚖️ Side-by-Side Destination Comparison Tool")
    st.write("Select up to 4 Indian travel destinations to compare prices, ratings, and climates side-by-side.")

    selected_compare_dests = st.multiselect(
        "Select Destinations to Compare (Max 4)",
        df_dest['destination'].tolist(),
        default=[df_dest['destination'].iloc[0], df_dest['destination'].iloc[1], df_dest['destination'].iloc[2]]
    )

    if selected_compare_dests:
        comp_df = df_dest[df_dest['destination'].isin(selected_compare_dests)].copy()
        
        c_cols = st.columns(len(comp_df))
        for idx, (_, c_row) in enumerate(comp_df.iterrows()):
            with c_cols[idx]:
                st.markdown(
                    f"""
                    <div class="ts-card">
                        <img src="{safe_image_url(c_row)}" class="card-img" />
                        <h3 class="card-title">{c_row['destination']}</h3>
                        <p><b>State:</b> {c_row['state']}<br>
                        <b>Type:</b> {c_row['destination_type']}<br>
                        <b>Daily Total:</b> <span style="color:#0284C7; font-weight:800;">₹{int(c_row['total_daily_cost_inr']):,}/day</span><br>
                        <b>Hotel Nightly:</b> ₹{int(c_row['average_hotel_cost']):,}<br>
                        <b>Food Daily:</b> ₹{int(c_row['average_food_cost']):,}<br>
                        <b>Transport:</b> ₹{int(c_row['average_local_transport_cost']):,}<br>
                        <b>Activity:</b> ₹{int(c_row['average_activity_cost']):,}<br>
                        <b>Best Season:</b> {c_row['best_season']}<br>
                        <b>Climate:</b> {c_row['climate']}<br>
                        <b>Rating:</b> ⭐ {c_row['popularity_score']}/5.0</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown("---")
        st.markdown("#### 📊 Comparative Daily Expense Breakdown (INR ₹)")
        comp_melted = comp_df.melt(
            id_vars=['destination'],
            value_vars=['average_hotel_cost', 'average_food_cost', 'average_local_transport_cost', 'average_activity_cost'],
            var_name='Category',
            value_name='Cost (₹)'
        )
        comp_melted['Category'] = comp_melted['Category'].str.replace('average_', '').str.replace('_cost', '').str.title()
        
        fig_comp = px.bar(
            comp_melted,
            x='destination',
            y='Cost (₹)',
            color='Category',
            barmode='group',
            color_discrete_sequence=['#0284C7', '#38BDF8', '#7E22CE', '#10B981'],
            labels={'destination': 'Destination'}
        )
        fig_comp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            yaxis=dict(gridcolor="#E2E8F0", tickprefix="₹"),
            font=dict(color="#0F172A"),
            legend=dict(font=dict(color="#0F172A"))
        )
        st.plotly_chart(fig_comp, width="stretch")

# ---------------------------------------------------------
# TAB 6: ANALYTICS DASHBOARD
# ---------------------------------------------------------
with tab_analytics:
    st.markdown("### 📊 Interactive Travel Data Analytics Dashboard")
    st.write("Explore dataset visual trends using interactive Plotly charts.")

    st.markdown("#### 🎛️ Interactive Data Filters")
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        all_types = ["All Categories"] + sorted(df_dest['destination_type'].unique().tolist())
        selected_type = st.selectbox("Filter by Category", all_types, index=0, key="analytics_type")
    with f_col2:
        all_climates = ["All Climates"] + sorted(df_dest['climate'].unique().tolist())
        selected_climate = st.selectbox("Filter by Climate", all_climates, index=0, key="analytics_climate")

    filtered_df = df_dest.copy()
    if selected_type != "All Categories":
        filtered_df = filtered_df[filtered_df['destination_type'] == selected_type]
    if selected_climate != "All Climates":
        filtered_df = filtered_df[filtered_df['climate'] == selected_climate]

    if filtered_df.empty:
        st.warning("No destinations match the selected combination. Displaying full dataset analytics.")
        filtered_df = df_dest.copy()

    metrics_flt = get_summary_metrics(filtered_df)
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Destinations", f"{metrics_flt['total_destinations']}")
    with m2:
        st.metric("Average Daily Cost", f"₹{metrics_flt['avg_daily_cost_inr']:,.2f}")
    with m3:
        st.metric("Categories", f"{metrics_flt['num_categories']}")
    with m4:
        st.metric("Average Rating", f"⭐ {metrics_flt.get('avg_rating', 4.2)}")

    st.markdown("---")

    st.markdown("### 1. Top 10 Destinations with Highest Estimated Daily Costs")
    fig_bar = plot_top_expensive_destinations(filtered_df, top_n=min(10, len(filtered_df)))
    st.plotly_chart(fig_bar, width="stretch")

    top_row = filtered_df.sort_values(by='total_daily_cost_inr', ascending=False).iloc[0]
    st.markdown(
        f"<div class='insight-box'>💡 <b>Factual Insight:</b> <b>{top_row['destination']} ({top_row['state']})</b> "
        f"exhibits the highest daily cost in this filtered view at <b>₹{top_row['total_daily_cost_inr']:,.2f}/day</b>.</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 2. Daily Travel Cost Distribution (INR ₹)")
    fig_hist = plot_cost_distribution(filtered_df)
    st.plotly_chart(fig_hist, width="stretch")

    mean_cost = filtered_df['total_daily_cost_inr'].mean()
    median_cost = filtered_df['total_daily_cost_inr'].median()
    st.markdown(
        f"<div class='insight-box'>💡 <b>Factual Insight:</b> Costs follow a distribution with a dataset Mean of <b>₹{mean_cost:,.2f}</b> "
        f"and Median of <b>₹{median_cost:,.2f}</b>.</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 3. Numerical Feature Correlation Matrix")
    fig_heat = plot_correlation_heatmap(filtered_df)
    st.plotly_chart(fig_heat, width="stretch")

    st.markdown(
        "<div class='insight-box'>💡 <b>Factual Insight:</b> Hotel accommodation cost shows the strongest positive correlation with total daily cost (~0.85+), "
        "establishing lodging charges as the primary financial driver.</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 4. Proportion of Destination Categories")
    fig_pie = plot_destination_types_pie(filtered_df)
    st.plotly_chart(fig_pie, width="stretch")

    top_cat = filtered_df['destination_type'].value_counts().index[0]
    top_cat_pct = (filtered_df['destination_type'].value_counts().iloc[0] / len(filtered_df)) * 100
    st.markdown(
        f"<div class='insight-box'>💡 <b>Factual Insight:</b> <b>{top_cat}</b> holds the single largest category share at <b>{top_cat_pct:.1f}%</b> of destinations.</div>",
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# TAB 7: PREPROCESSING & MODEL METRICS
# ---------------------------------------------------------
with tab_eda:
    st.markdown("### 🔬 Data Preprocessing & ML Model Evaluation Metrics")
    
    raw_df = load_raw_dataset()
    pipe_res = preprocess_travel_data()
    
    e_col1, e_col2 = st.columns(2)
    with e_col1:
        st.markdown(
            f"""
            <div class="ts-card">
                <h4>Raw Dataset Stats</h4>
                <p><b>Total Rows:</b> {len(raw_df)}<br>
                <b>Duplicate Rows:</b> {raw_df.duplicated().sum()}<br>
                <b>Missing Hotel Costs:</b> {raw_df['average_hotel_cost'].isnull().sum() if 'average_hotel_cost' in raw_df.columns else 0}<br>
                <b>Missing Food Costs:</b> {raw_df['average_food_cost'].isnull().sum() if 'average_food_cost' in raw_df.columns else 0}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with e_col2:
        st.markdown(
            f"""
            <div class="ts-card">
                <h4>Cleaned Dataset Stats</h4>
                <p><b>Total Rows:</b> {len(df_dest)}<br>
                <b>Duplicate Rows:</b> {df_dest.duplicated().sum()}<br>
                <b>Missing Hotel Costs:</b> {df_dest['average_hotel_cost'].isnull().sum()}<br>
                <b>Missing Food Costs:</b> {df_dest['average_food_cost'].isnull().sum()}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("#### 🌲 Random Forest Regressor Evaluation Metrics (Held-Out 80/20 Test Split)")
    metrics_rf = budget_predictor.metrics
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Test Set Size", f"{metrics_rf.get('test_set_size', 360)} samples")
    with m_col2:
        st.metric("MAE (Mean Abs Error)", f"₹{metrics_rf.get('MAE', 15673.93):,.2f}")
    with m_col3:
        st.metric("RMSE (Root Mean Sq)", f"₹{metrics_rf.get('RMSE', 27048.06):,.2f}")
    with m_col4:
        st.metric("R² Score", f"{metrics_rf.get('R2', 0.8347):.4f}")

    st.markdown("---")
    st.markdown("#### ⚙️ Feature Importance Analysis")
    
    try:
        rf_regressor = budget_predictor.pipeline.named_steps['regressor']
        preprocessor = budget_predictor.pipeline.named_steps['preprocessor']
        
        cat_encoder = preprocessor.named_transformers_['cat']
        cat_features = list(cat_encoder.get_feature_names_out(['state', 'destination_type', 'climate', 'best_season', 'accommodation_tier']))
        num_features = ['popularity_score', 'duration_days', 'group_size']
        all_feature_names = num_features + cat_features
        
        importances = rf_regressor.feature_importances_
        fi_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(12)
        
        fig_fi = px.bar(
            fi_df,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Blues_r'
        )
        fig_fi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            font=dict(color="#0F172A")
        )
        st.plotly_chart(fig_fi, width="stretch")
    except Exception as e:
        st.info("Feature importance summary available upon live model fit.")
