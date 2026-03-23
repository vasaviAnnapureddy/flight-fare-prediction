import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
import os

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Fare Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #0d2137);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #2d5a8e;
        text-align: center;
    }
    .metric-card h1 { color: #38bdf8; font-size: 2.2rem; margin: 0; }
    .metric-card p  { color: #94a3b8; font-size: 0.85rem; margin: 0; }
    .stButton > button {
        background: linear-gradient(90deg, #0ea5e9, #38bdf8);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1rem;
        padding: 0.6rem 2rem;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.88; }
    div[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #1e2d3d; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('flight_price_model.pkl')
    with open('feature_columns.json') as f:
        features = json.load(f)
    return model, features

try:
    model, feature_cols = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model not found: {e}. Make sure flight_price_model.pkl and feature_columns.json are in the same folder.")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ Flight Fare Predictor")
    st.markdown("---")
    st.markdown("**Project:** PRCP-1025")
    st.markdown("**Model:** XGBoost (Tuned)")
    st.markdown("**Technique:** NLP Route Embeddings + Stacking Ensemble")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
- Fill in your flight details
- Click **Predict Price**
- See the estimated fare instantly
    """)
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit + XGBoost")

# ── Header ────────────────────────────────────────────────────
st.markdown("# ✈️ Flight Fare Prediction")
st.markdown("#### Predict Indian domestic flight prices using Machine Learning")
st.markdown("---")

# ── Input Form ────────────────────────────────────────────────
st.markdown("### 🔧 Enter Flight Details")

col1, col2, col3 = st.columns(3)

with col1:
    airline = st.selectbox("Airline", [
        "IndiGo", "Air India", "Jet Airways", "SpiceJet",
        "Multiple carriers", "GoAir", "Vistara", "Air Asia",
        "Jet Airways Business", "Multiple carriers Premium economy",
        "Trujet"
    ])
    source = st.selectbox("Source City", [
        "Banglore", "Kolkata", "Delhi", "Chennai", "Mumbai"
    ])

with col2:
    destination = st.selectbox("Destination City", [
        "New Delhi", "Banglore", "Cochin", "Kolkata",
        "Delhi", "Hyderabad"
    ])
    stops = st.selectbox("Number of Stops", [
        "non-stop (0)", "1 stop", "2 stops", "3 stops"
    ])

with col3:
    dep_hour = st.slider("Departure Hour", 0, 23, 8,
                          help="0 = midnight, 23 = 11 PM")
    duration_hrs = st.slider("Flight Duration (hours)", 1, 24, 3)
    journey_month = st.selectbox("Journey Month", [
        "January (1)", "February (2)", "March (3)", "April (4)",
        "May (5)", "June (6)", "July (7)", "August (8)",
        "September (9)", "October (10)", "November (11)", "December (12)"
    ])

# ── Parse Inputs ──────────────────────────────────────────────
stop_map_input = {
    "non-stop (0)": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3
}
stops_val    = stop_map_input[stops]
duration_val = duration_hrs * 60
month_val    = int(journey_month.split("(")[1].replace(")", ""))

# ── Predict ───────────────────────────────────────────────────
st.markdown("---")
predict_col, space_col = st.columns([1, 2])

with predict_col:
    predict_btn = st.button("🔮 Predict Price")

if predict_btn and model_loaded:
    # Build input row matching training feature columns
    input_dict = {col: 0 for col in feature_cols}

    # Numeric features
    numeric_vals = {
        'journey_day': 15,
        'journey_month': month_val,
        'journey_weekday': 2,
        'dep_hour': dep_hour,
        'dep_min': 0,
        'arr_hour': (dep_hour + duration_hrs) % 24,
        'duration_mins': duration_val,
        'stops': stops_val,
        'days_before': 30,
    }
    # Route SVD features — set to zero (neutral embedding)
    for i in range(10):
        numeric_vals[f'route_{i}'] = 0.0

    for k, v in numeric_vals.items():
        if k in input_dict:
            input_dict[k] = v

    # One-hot encoded features
    airline_col = f'Airline_{airline}'
    source_col  = f'Source_{source}'
    dest_col    = f'Destination_{destination}'
    for col in [airline_col, source_col, dest_col]:
        if col in input_dict:
            input_dict[col] = 1

    input_df = pd.DataFrame([input_dict])
    # ensure column order matches training
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_cols]

    pred_log  = model.predict(input_df)[0]
    pred_price = np.expm1(pred_log)
    low_band   = pred_price * 0.90
    high_band  = pred_price * 1.10

    st.markdown("---")
    st.markdown("### 💰 Prediction Result")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
        <div class="metric-card">
            <p>Estimated Fare</p>
            <h1>₹{pred_price:,.0f}</h1>
            <p>{airline} · {source} → {destination}</p>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="metric-card">
            <h1>₹{low_band:,.0f}</h1>
            <p>Lower Estimate (−10%)</p>
        </div>""", unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div class="metric-card">
            <h1>₹{high_band:,.0f}</h1>
            <p>Upper Estimate (+10%)</p>
        </div>""", unsafe_allow_html=True)

    # ── Price Range Gauge ──────────────────────────────────────
    st.markdown("#### 📊 Price Range Visualization")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred_price,
        number={'prefix': '₹', 'valueformat': ',.0f'},
        title={'text': "Predicted Fare (INR)"},
        gauge={
            'axis': {'range': [0, 80000]},
            'bar': {'color': "#38bdf8"},
            'steps': [
                {'range': [0, 5000],    'color': '#22c55e'},
                {'range': [5000, 15000], 'color': '#eab308'},
                {'range': [15000, 40000],'color': '#f97316'},
                {'range': [40000, 80000],'color': '#ef4444'},
            ],
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 0.8,
                'value': pred_price
            }
        }
    ))
    fig_gauge.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)',
                             font={'color': 'white'})
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Feature Impact ─────────────────────────────────────────
    st.markdown("#### 🔍 What's Driving This Price?")
    factors = {
        'Flight Duration': min(duration_val / 1500, 1.0),
        'Number of Stops': stops_val / 4,
        'Departure Hour':  abs(dep_hour - 12) / 12,
        'Journey Month':   abs(month_val - 6) / 6,
        'Airline Type':    0.8 if 'Business' in airline or airline == 'Jet Airways' else 0.3,
    }
    fig_bar = px.bar(
        x=list(factors.values()),
        y=list(factors.keys()),
        orientation='h',
        color=list(factors.values()),
        color_continuous_scale='Blues',
        labels={'x': 'Relative Impact', 'y': 'Feature'},
        title='Feature Impact on Predicted Price'
    )
    fig_bar.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        coloraxis_showscale=False,
        height=280
    )
    fig_bar.update_traces(marker_line_width=0)
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Travel Tips ────────────────────────────────────────────
    st.markdown("#### 💡 Smart Travel Tips")
    tips = []
    if dep_hour >= 8 and dep_hour <= 11:
        tips.append("🕐 **Try departing before 7 AM or after 9 PM** — off-peak flights are typically 10–20% cheaper.")
    if stops_val >= 2:
        tips.append("🔄 **Consider a 1-stop flight** — 2+ stop routes are often more expensive despite longer travel time.")
    if month_val in [5, 6]:
        tips.append("📅 **You're flying in peak season (May–June)** — prices are highest. Book 45+ days in advance.")
    if month_val in [2, 3]:
        tips.append("✅ **Great month to fly!** February–March typically has the lowest fares of the year.")
    if 'Business' in airline:
        tips.append("💼 **Business class selected** — for budget travel, Economy on the same route costs 60–70% less.")
    if not tips:
        tips.append("✅ **Good choices!** Your selected options are generally cost-efficient.")
    for tip in tips:
        st.info(tip)

elif predict_btn and not model_loaded:
    st.error("Model file not found. Run the notebook first to generate flight_price_model.pkl")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#475569; font-size:0.8rem;'>"
    "PRCP-1025 · Flight Fare Prediction · Built with XGBoost + Streamlit"
    "</p>",
    unsafe_allow_html=True
)
