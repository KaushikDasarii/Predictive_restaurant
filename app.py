# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

# ============ Page Setup ============
st.set_page_config(
    page_title="Restaurant Recommendation System",
    layout="wide",
    page_icon="🍽️",
    initial_sidebar_state="expanded"
)

st.title("🍽️ Restaurant Recommender Engine")
st.markdown("Predict whether a customer is likely to order from a restaurant based on past behavior, demographics, and location.")

# ============ Load Model & Data ============
@st.cache_resource
def load_model():
    return joblib.load("xgboost_model.pkl")

@st.cache_data
def load_data():
    customers = pd.read_csv("train_customers_clean.csv")
    locations = pd.read_csv("train_locations_clean.csv")
    vendors = pd.read_csv("vendors_clean.csv")
    # 🔧 FIX: ensure location number is integer
    locations['location_number'] = locations['location_number'].astype(int)
    return customers, locations, vendors

model = load_model()
customers_df, locations_df, vendors_df = load_data()

# ============ Dynamic Input Selection ============
st.sidebar.header("🔍 Enter Query Details")

# 1. Select valid customer ID
cid_list = sorted(customers_df['customer_id'].dropna().unique())
cid = st.sidebar.selectbox("Select Customer ID (CID)", cid_list)

# 2. Filter location numbers for selected customer (int-safe)
loc_options = sorted(locations_df[locations_df['customer_id'] == cid]['location_number'].unique())
loc_num = st.sidebar.selectbox("Select Location Number", loc_options)

# 3. Vendor selection
vendor_ids = sorted(vendors_df['id'].dropna().unique())
vendor_id = st.sidebar.selectbox("Select Vendor ID", vendor_ids)

submit = st.sidebar.button("🔮 Predict")

# ============ Feature Engineering ============
def prepare_features(cid, loc_num, vendor_id):
    df = pd.DataFrame([{'CID': cid, 'LOC_NUM': loc_num, 'VENDOR': vendor_id}])

    # Customer features
    cust_feat = customers_df[customers_df['customer_id'] == cid][['customer_id', 'gender', 'age', 'language', 'account_age_days']]
    if cust_feat.empty: return None
    cust_feat.columns = ['CID', 'gender', 'age', 'language', 'account_age_days']
    df = df.merge(cust_feat, on='CID', how='left')

    # Location features
    loc_feat = locations_df[(locations_df['customer_id'] == cid) & (locations_df['location_number'] == loc_num)]
    if loc_feat.empty: return None
    loc_feat = loc_feat[['customer_id', 'location_number', 'location_type', 'latitude', 'longitude']]
    loc_feat.columns = ['CID', 'LOC_NUM', 'location_type', 'cust_lat', 'cust_long']
    df = df.merge(loc_feat, on=['CID', 'LOC_NUM'], how='left')

    # Vendor features
    ven_feat = vendors_df[vendors_df['id'] == vendor_id][['id', 'vendor_tag_name', 'latitude', 'longitude']]
    if ven_feat.empty: return None
    ven_feat.columns = ['VENDOR', 'vendor_tag', 'vendor_lat', 'vendor_long']
    df = df.merge(ven_feat, on='VENDOR', how='left')

    # Distance
    df['distance'] = np.sqrt(
        (df['cust_lat'] - df['vendor_lat'])**2 +
        (df['cust_long'] - df['vendor_long'])**2
    )

    # Fill missing numerics
    df['age'] = df['age'].fillna(-1)
    df['account_age_days'] = df['account_age_days'].fillna(-1)
    df['distance'] = df['distance'].fillna(df['distance'].mean())

    # Encode categoricals
    for col in ['gender', 'language', 'location_type', 'vendor_tag']:
        df[col] = df[col].fillna("unknown")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df

# ============ Prediction ============

if submit:
    with st.spinner("🔍 Running prediction..."):
        features = prepare_features(cid, loc_num, vendor_id)

        if features is None or features.isnull().values.any():
            st.error("⚠️ Missing or invalid data for this combination.")
        else:
            X = features.drop(columns=['CID', 'LOC_NUM', 'VENDOR'])
            dtest = xgb.DMatrix(X)
            prob = model.predict(dtest)[0]
            label = "✅ Likely to Order" if prob >= 0.5 else "❌ Unlikely to Order"

            # === Output Summary ===
            st.success(f"**Prediction:** {label}")
            st.metric("Probability", f"{prob*100:.2f}%")

            # === Gauge Chart ===
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': "Order Likelihood (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightcoral"},
                        {'range': [50, 100], 'color': "lightgreen"}]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

            # === Feature Table ===
            with st.expander("📋 Show Feature Details"):
                st.dataframe(features)
