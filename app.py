
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from predict import CropAndYieldPredictor, get_crop_agronomic_profiles

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AgroAI - Crop Recommendation & Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #2e7d32;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 20px;
    }
    .metric-box {
        background-color: #f8fafc;
        border-left: 5px solid #2e7d32;
        border-radius: 8px;
        padding: 15px 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        margin-bottom: 12px;
    }
    .crop-badge {
        display: inline-block;
        background: linear-gradient(135deg, #2e7d32, #43a047);
        color: white;
        padding: 8px 18px;
        border-radius: 20px;
        font-size: 1.4rem;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border: 1px solid #a5d6a7;
        border-radius: 12px;
        padding: 20px;
        color: #1b5e20;
        margin-bottom: 20px;
    }
    .yield-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border: 1px solid #90caf9;
        border-radius: 12px;
        padding: 20px;
        color: #0d47a1;
        margin-bottom: 20px;
    }
    .tag {
        background-color: #e2e8f0;
        color: #2d3748;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Helper to Load Predictor
@st.cache_resource
def load_engine():
    try:
        return CropAndYieldPredictor(models_dir='models')
    except Exception as e:
        st.error(f"Error loading machine learning models: {e}")
        st.info("Ensure models are trained by running `python train_crop_model.py` and `python train_yield_model.py`.")
        return None

predictor = load_engine()
crop_profiles = get_crop_agronomic_profiles()

# Regional Presets Dictionary
PRESETS = {
    "Select a Preset...": None,
    "🌊 Tropical Wet Basin (High Rain / High Humidity)": {
        'N': 85, 'P': 48, 'K': 40, 'temp': 24.0, 'hum': 84.0, 'ph': 6.4, 'rain': 240.0, 'area': 3.0
    },
    "🍏 Temperate Highland Orchard (High K / P)": {
        'N': 22, 'P': 130, 'K': 200, 'temp': 22.0, 'hum': 92.0, 'ph': 6.0, 'rain': 112.0, 'area': 2.5
    },
    "🏜️ Semi-Arid Pulse Belt (Low Moisture / Alkaline)": {
        'N': 40, 'P': 65, 'K': 80, 'temp': 19.0, 'hum': 17.0, 'ph': 7.3, 'rain': 80.0, 'area': 4.0
    },
    "🍌 Humid Coastal Plantation (High Nitrogen & Moisture)": {
        'N': 100, 'P': 80, 'K': 50, 'temp': 27.5, 'hum': 80.0, 'ph': 6.0, 'rain': 105.0, 'area': 5.0
    },
    "🌾 Alluvial Cereal Plains (High Nitrogen / Moderate Rain)": {
        'N': 80, 'P': 50, 'K': 20, 'temp': 23.0, 'hum': 65.0, 'ph': 6.5, 'rain': 85.0, 'area': 6.0
    }
}

# Sidebar Navigation
st.sidebar.title("🌾 AgroAI Navigation")
app_mode = st.sidebar.radio(
    "Select System Module:",
    [
        "🌱 End-to-End Precision Predictor",
        "🛠️ User-Defined Prediction & Batch Tools",
        "🌾 Crop Recommender (Stage 1)",
        "📈 Yield & Production Forecast (Stage 2)",
        "📊 Analytics & Model Benchmarks",
        "📚 Academic Viva & ML Theory Hub"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Status")
if predictor:
    st.sidebar.success("✅ Models Loaded & Active")
    st.sidebar.caption(f"Classifier: **{predictor.crop_metadata.get('best_model', 'Random Forest')}**")
    st.sidebar.caption(f"Regressor: **{predictor.yield_metadata.get('best_model', 'Random Forest Regressor')}**")
else:
    st.sidebar.error("❌ Models Not Loaded")

st.sidebar.markdown("---")
st.sidebar.caption("🌾 Intelligent Agricultural Decision Support System\nDeep Learning & Machine Learning Project")

# ==========================================
# MODULE 1: END-TO-END PRECISION PREDICTOR
# ==========================================
if app_mode == "🌱 End-to-End Precision Predictor":
    st.markdown('<div class="main-header">🌾 End-to-End Intelligent Agricultural Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-stage pipeline: Recommends the optimal crop variety via machine learning classification and forecasts harvest yield in Tonnes, Quintals, and Kilograms.</div>', unsafe_allow_html=True)

    # Preset selection
    st.markdown("#### ⚡ Quick Regional Agro-Climatic Presets")
    selected_preset = st.selectbox("Load pre-configured soil & meteorological profile:", list(PRESETS.keys()))

    preset_vals = PRESETS.get(selected_preset, None)

    # Form inputs
    with st.form("precision_form"):
        st.markdown("### 🧪 Soil Macronutrients & Geochemistry")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            n_val = st.number_input("Nitrogen (N) [kg/ha]", min_value=0, max_value=200, value=preset_vals['N'] if preset_vals else 85, help="Soil available Nitrogen ratio")
        with c2:
            p_val = st.number_input("Phosphorus (P) [kg/ha]", min_value=0, max_value=200, value=preset_vals['P'] if preset_vals else 48, help="Soil available Phosphorus ratio")
        with c3:
            k_val = st.number_input("Potassium (K) [kg/ha]", min_value=0, max_value=250, value=preset_vals['K'] if preset_vals else 40, help="Soil available Potassium ratio")
        with c4:
            ph_val = st.number_input("Soil pH Level", min_value=3.0, max_value=10.0, value=preset_vals['ph'] if preset_vals else 6.4, step=0.1, help="Soil pH (0-14 scale)")

        st.markdown("### ⛅ Atmospheric & Climatic Variables")
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            temp_val = st.number_input("Temperature (°C)", min_value=5.0, max_value=50.0, value=preset_vals['temp'] if preset_vals else 24.0, step=0.5)
        with c6:
            hum_val = st.number_input("Relative Humidity (%)", min_value=10.0, max_value=100.0, value=preset_vals['hum'] if preset_vals else 84.0, step=1.0)
        with c7:
            rain_val = st.number_input("Rainfall (mm)", min_value=10.0, max_value=500.0, value=preset_vals['rain'] if preset_vals else 240.0, step=5.0)
        with c8:
            area_val = st.number_input("Farm Land Area (ha)", min_value=0.1, max_value=100.0, value=preset_vals['area'] if preset_vals else 3.0, step=0.5)

        submit_btn = st.form_submit_button("🚀 Run Intelligent Decision Pipeline", use_container_width=True)

    if submit_btn:
        if not predictor:
            st.error("Model engine is not ready.")
        else:
            with st.spinner("Analyzing soil chemistry and climate matrices..."):
                results = predictor.predict_all(
                    n=n_val, p=p_val, k=k_val,
                    temp=temp_val, hum=hum_val, ph=ph_val, rain=rain_val,
                    area_ha=area_val
                )

            rec = results['stage1_recommendation']
            yld = results['stage2_yield_forecast']
            recommended_crop = rec['recommended_crop']

            st.markdown("---")
            st.markdown("## 📊 Intelligent Prediction Results")

            col_left, col_right = st.columns([1.1, 0.9])

            with col_left:
                # Stage 1 Card
                st.markdown(f"""
                <div class="recommendation-card">
                    <h3 style="margin-top:0; color:#1b5e20;">🌱 Stage 1: Recommended Optimal Crop</h3>
                    <div style="margin: 10px 0;">
                        <span class="crop-badge">{recommended_crop}</span>
                        <span style="font-size:1.1rem; margin-left:15px; font-weight:600;">Confidence: {rec['confidence_percent']:.1f}%</span>
                    </div>
                    <p style="margin-top:10px; font-size:0.95rem; color:#2e7d32;">
                        The multi-class classification model identified <b>{recommended_crop.upper()}</b> as the physiologically optimal crop variety matching the input soil chemistry and climate parameters.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Stage 2 Card
                st.markdown(f"""
                <div class="yield-card">
                    <h3 style="margin-top:0; color:#0d47a1;">📈 Stage 2: Yield & Production Forecast</h3>
                    <div style="display:flex; justify-content:space-between; flex-wrap:wrap;">
                        <div>
                            <div style="font-size:0.9rem; color:#1565c0;">Yield Efficiency</div>
                            <div style="font-size:1.8rem; font-weight:bold; color:#0d47a1;">{yld['yield_tonnes_per_ha']} <span style="font-size:1rem;">Tonnes/Ha</span></div>
                        </div>
                        <div>
                            <div style="font-size:0.9rem; color:#1565c0;">Total Harvest ({area_val} ha)</div>
                            <div style="font-size:1.8rem; font-weight:bold; color:#0d47a1;">{yld['total_production_tonnes']} <span style="font-size:1rem;">Tonnes</span></div>
                        </div>
                    </div>
                    <hr style="border-color:#bbdefb; margin:12px 0;">
                    <div style="display:flex; justify-content:space-around; text-align:center;">
                        <div><b>{yld['total_production_quintals']}</b> Quintals</div>
                        <div>|</div>
                        <div><b>{yld['total_production_kg']:,.0f}</b> Kilograms</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_right:
                # Top Alternatives Chart
                st.markdown("#### 🏆 Top Crop Alternatives Probability")
                top_df = pd.DataFrame(rec['top_candidates'])
                fig_bar = px.bar(
                    top_df, x='confidence', y='crop',
                    orientation='h',
                    text='confidence',
                    color='confidence',
                    color_continuous_scale='Greens',
                    labels={'confidence': 'Model Confidence (%)', 'crop': 'Crop'}
                )
                fig_bar.update_layout(height=240, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
                fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig_bar, use_container_width=True)

                # Radar chart of soil nutrients vs optimal crop profile
                if recommended_crop in crop_profiles:
                    prof = crop_profiles[recommended_crop]
                    categories = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)', 'Temperature', 'Humidity', 'Rainfall']
                    
                    # Normalize values to 0-100 scale for radar comparison
                    user_norm = [
                        min(100, (n_val / 140) * 100),
                        min(100, (p_val / 145) * 100),
                        min(100, (k_val / 205) * 100),
                        min(100, (temp_val / 40) * 100),
                        min(100, (hum_val / 100) * 100),
                        min(100, (rain_val / 300) * 100)
                    ]
                    ideal_norm = [
                        min(100, (prof['N'][0] / 140) * 100),
                        min(100, (prof['P'][0] / 145) * 100),
                        min(100, (prof['K'][0] / 205) * 100),
                        min(100, (prof['temp'][0] / 40) * 100),
                        min(100, (prof['hum'][0] / 100) * 100),
                        min(100, (prof['rain'][0] / 300) * 100)
                    ]

                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(r=user_norm, theta=categories, fill='toself', name='Current Farm Profile', line_color='#e67e22'))
                    fig_radar.add_trace(go.Scatterpolar(r=ideal_norm, theta=categories, fill='toself', name=f'Ideal {recommended_crop.title()}', line_color='#27ae60'))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        showlegend=True,
                        height=260,
                        margin=dict(l=20, r=20, t=20, b=20)
                    )
                    st.markdown("#### 🎯 Agronomic Profile Radar Comparison")
                    st.plotly_chart(fig_radar, use_container_width=True)

# ==========================================
# MODULE: USER-DEFINED PREDICTION & BATCH TOOLS
# ==========================================
elif app_mode == "🛠️ User-Defined Prediction & Batch Tools":
    st.markdown('<div class="main-header">🛠️ User-Defined Custom Prediction & Batch Tools</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Input custom soil & climate measurements, process batch CSV datasets, or define new custom crop profiles.</div>', unsafe_allow_html=True)

    ud_tab1, ud_tab2, ud_tab3 = st.tabs([
        "✍️ User-Defined Single Prediction",
        "📂 Batch User-Defined CSV Predictor",
        "➕ Add Custom / User-Defined Crop"
    ])

    with ud_tab1:
        st.markdown("### 📝 Enter Custom Soil Chemistry & Environmental Measurements")
        
        ud_col1, ud_col2 = st.columns([1, 1])
        with ud_col1:
            ud_n = st.number_input("User-Defined Nitrogen (N) [kg/ha]", 0.0, 300.0, 92.0, step=1.0)
            ud_p = st.number_input("User-Defined Phosphorus (P) [kg/ha]", 0.0, 300.0, 46.0, step=1.0)
            ud_k = st.number_input("User-Defined Potassium (K) [kg/ha]", 0.0, 300.0, 42.0, step=1.0)
            ud_ph = st.slider("User-Defined Soil pH Level", 3.0, 10.0, 6.6, 0.1)

        with ud_col2:
            ud_temp = st.number_input("User-Defined Temperature (°C)", 0.0, 60.0, 24.5, step=0.5)
            ud_hum = st.number_input("User-Defined Relative Humidity (%)", 5.0, 100.0, 81.0, step=1.0)
            ud_rain = st.number_input("User-Defined Rainfall (mm)", 0.0, 1000.0, 230.0, step=5.0)
            ud_area = st.number_input("User-Defined Farm Area (ha)", 0.1, 1000.0, 3.5, step=0.5)

        # Stoichiometry analysis
        npk_total = ud_n + ud_p + ud_k + 1e-6
        n_ratio = round((ud_n / npk_total) * 100, 1)
        p_ratio = round((ud_p / npk_total) * 100, 1)
        k_ratio = round((ud_k / npk_total) * 100, 1)

        st.caption(f"🌿 **User NPK Ratio**: N ({n_ratio}%) : P ({p_ratio}%) : K ({k_ratio}%) | Ideal Benchmark ~ 4 : 2 : 1")

        run_ud = st.button("🚀 Run User-Defined Inference", key="btn_ud_single", use_container_width=True)

        if run_ud and predictor:
            ud_res = predictor.predict_all(
                n=ud_n, p=ud_p, k=ud_k,
                temp=ud_temp, hum=ud_hum, ph=ud_ph, rain=ud_rain,
                area_ha=ud_area
            )

            ud_rec = ud_res['stage1_recommendation']
            ud_yld = ud_res['stage2_yield_forecast']

            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.success(f"### Recommended Crop: **{ud_rec['recommended_crop'].upper()}**")
                st.metric("Model Confidence", f"{ud_rec['confidence_percent']:.1f}%")
                st.write("**Top Candidates:**")
                for cand in ud_rec['top_candidates']:
                    st.write(f"- {cand['crop'].title()}: `{cand['confidence']}%`")

            with r_col2:
                st.info(f"### Estimated Yield: **{ud_yld['yield_tonnes_per_ha']} Tonnes/Ha**")
                st.metric("Total Farm Harvest", f"{ud_yld['total_production_tonnes']} Tonnes", f"{ud_yld['total_production_quintals']} Quintals")
                st.write(f"Harvest in Kilograms: **{ud_yld['total_production_kg']:,.0f} kg**")

            # Downloadable Report
            report_dict = {
                "user_inputs": {
                    "N": ud_n, "P": ud_p, "K": ud_k,
                    "temperature": ud_temp, "humidity": ud_hum,
                    "ph": ud_ph, "rainfall": ud_rain, "area_ha": ud_area
                },
                "stage1_crop_recommendation": ud_rec,
                "stage2_yield_forecast": ud_yld
            }
            json_str = json.dumps(report_dict, indent=4)
            st.download_button(
                label="📥 Download User Prediction Report (JSON)",
                data=json_str,
                file_name="user_defined_prediction_report.json",
                mime="application/json"
            )

    with ud_tab2:
        st.markdown("### 📂 Upload User-Defined Batch CSV Dataset")
        st.write("Upload a CSV file containing columns: `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`, and `Area_ha`.")

        sample_csv_data = (
            "N,P,K,temperature,humidity,ph,rainfall,Area_ha\n"
            "85,48,40,24.0,84.0,6.4,240.0,3.0\n"
            "20,130,200,22.0,92.0,6.0,110.0,2.5\n"
            "40,65,80,19.0,16.0,7.2,80.0,5.0\n"
            "120,45,20,24.5,78.0,6.8,80.0,4.0\n"
            "100,82,50,27.4,80.4,6.0,105.0,2.0\n"
            "78,48,20,22.4,65.1,6.2,85.0,6.0\n"
        )

        c_up1, c_up2 = st.columns([1, 1])
        with c_up1:
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        with c_up2:
            st.write("Or test with pre-built user-defined sample:")
            use_sample = st.button("🧪 Load Sample Multi-Farm Batch CSV", use_container_width=True)

        batch_df = None
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"Loaded user file with {len(batch_df)} records.")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
        elif use_sample:
            import io
            batch_df = pd.read_csv(io.StringIO(sample_csv_data))
            st.info(f"Loaded sample multi-farm batch with {len(batch_df)} records.")

        if batch_df is not None and predictor:
            st.markdown("#### 📋 Preview Input Data")
            st.dataframe(batch_df.head(10), use_container_width=True)

            if st.button("⚡ Process Batch Predictions", use_container_width=True):
                with st.spinner("Processing batch with Machine Learning models..."):
                    out_df = predictor.predict_batch(batch_df)

                st.markdown("### 📊 Batch Prediction Results")
                st.dataframe(out_df, use_container_width=True)

                # Batch summary metrics
                st.markdown("#### 📈 Aggregate Production Summary")
                sum_c1, sum_c2, sum_c3 = st.columns(3)
                with sum_c1:
                    st.metric("Total Land Area", f"{out_df['Area_ha'].sum():.2f} ha")
                with sum_c2:
                    st.metric("Total Forecasted Harvest", f"{out_df['Total_Production_Tonnes'].sum():.2f} Tonnes")
                with sum_c3:
                    st.metric("Total Harvest in Quintals", f"{out_df['Total_Production_Quintals'].sum():.1f} Q")

                # Crop distribution bar chart in batch
                fig_dist = px.histogram(out_df, x='Recommended_Crop', color='Recommended_Crop', title="Recommended Crop Variety Distribution Across Farms")
                st.plotly_chart(fig_dist, use_container_width=True)

                csv_bytes = out_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Batch Prediction Results (CSV)",
                    data=csv_bytes,
                    file_name="batch_user_defined_predictions.csv",
                    mime="text/csv"
                )

    with ud_tab3:
        st.markdown("### ➕ Register a Custom / User-Defined Crop Variety")
        st.write("Add your own custom crop to the agronomic engine by specifying its physiological requirements and baseline yield.")

        with st.form("custom_crop_form"):
            cc_name = st.text_input("Custom Crop Name", value="Dragon Fruit")
            
            c_c1, c_c2, c_c3 = st.columns(3)
            with c_c1:
                cc_n = st.number_input("Optimal Nitrogen (N) kg/ha", 0.0, 300.0, 70.0)
                cc_p = st.number_input("Optimal Phosphorus (P) kg/ha", 0.0, 300.0, 35.0)
                cc_k = st.number_input("Optimal Potassium (K) kg/ha", 0.0, 300.0, 45.0)
            with c_c2:
                cc_temp = st.number_input("Optimal Temp (°C)", 5.0, 50.0, 28.0)
                cc_hum = st.number_input("Optimal Humidity (%)", 10.0, 100.0, 70.0)
                cc_ph = st.number_input("Optimal Soil pH", 4.0, 9.0, 6.5)
            with c_c3:
                cc_rain = st.number_input("Optimal Rainfall (mm)", 10.0, 500.0, 90.0)
                cc_yield = st.number_input("Base Yield (Tonnes/Ha)", 0.5, 100.0, 12.5)

            add_crop_btn = st.form_submit_button("🌱 Register Custom Crop Variety", use_container_width=True)

        if add_crop_btn and predictor:
            prof = predictor.add_user_defined_crop(
                crop_name=cc_name,
                n=cc_n, p=cc_p, k=cc_k,
                temp=cc_temp, hum=cc_hum, ph=cc_ph, rain=cc_rain,
                base_yield=cc_yield
            )
            st.success(f"✅ Successfully registered custom crop: **{cc_name.title()}**!")
            st.json({
                "crop": cc_name.lower(),
                "ideal_n": cc_n, "ideal_p": cc_p, "ideal_k": cc_k,
                "ideal_temp": cc_temp, "ideal_humidity": cc_hum,
                "ideal_ph": cc_ph, "ideal_rainfall": cc_rain,
                "base_yield_tonnes_ha": cc_yield
            })

# ==========================================
# MODULE 2: CROP RECOMMENDER (STAGE 1)
# ==========================================
elif app_mode == "🌾 Crop Recommender (Stage 1)":
    st.markdown('<div class="main-header">🌾 Crop Suitability Recommender (Stage 1 Classification)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluate multi-class machine learning models across 22 crops with detailed confidence rankings.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🔬 Soil & Climate Inputs")
        n = st.slider("Nitrogen (N) in soil (kg/ha)", 0, 150, 40)
        p = st.slider("Phosphorus (P) in soil (kg/ha)", 0, 150, 68)
        k = st.slider("Potassium (K) in soil (kg/ha)", 0, 220, 80)
        temp = st.slider("Temperature (°C)", 10.0, 45.0, 19.0, 0.5)
        hum = st.slider("Relative Humidity (%)", 10.0, 100.0, 17.0, 0.5)
        ph = st.slider("Soil pH Level", 3.5, 9.5, 7.3, 0.1)
        rain = st.slider("Rainfall (mm)", 10.0, 350.0, 80.0, 5.0)

        calc_crop = st.button("🌱 Recommend Optimal Crop", use_container_width=True)

    with col2:
        st.markdown("### 🏆 Recommendation Insights")
        if calc_crop and predictor:
            res = predictor.predict_crop(n, p, k, temp, hum, ph, rain, top_k=6)
            rec_crop = res['recommended_crop']
            conf = res['confidence_percent']

            st.success(f"### Optimal Recommendation: **{rec_crop.upper()}** ({conf:.1f}% Confidence)")
            
            top_candidates = pd.DataFrame(res['top_candidates'])
            fig_top = px.bar(
                top_candidates, x='confidence', y='crop', orientation='h',
                color='confidence', color_continuous_scale='teal',
                text='confidence', labels={'confidence': 'Confidence %', 'crop': 'Crop Variety'}
            )
            fig_top.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_top.update_layout(title="Top 6 Crop Candidates by Classification Probability")
            st.plotly_chart(fig_top, use_container_width=True)

            st.markdown("#### 📋 Agronomic Assessment")
            st.info(f"The input conditions closely match the physiological requirements of **{rec_crop}**. Temperature ({temp}°C) and pH ({ph}) are within optimal vegetative tolerance thresholds.")

# ==========================================
# MODULE 3: YIELD & PRODUCTION FORECAST
# ==========================================
elif app_mode == "📈 Yield & Production Forecast (Stage 2)":
    st.markdown('<div class="main-header">📈 Crop Yield & Harvest Production Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Simulate yield efficiency (Tonnes/Ha) and total farm harvest for any specific crop variety.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("### 🚜 Farm & Crop Configuration")
        crops_list = sorted(list(crop_profiles.keys()))
        selected_crop = st.selectbox("Select Crop Variety:", crops_list, index=crops_list.index('rice') if 'rice' in crops_list else 0)
        area_ha = st.number_input("Farm Land Area (Hectares):", min_value=0.1, max_value=500.0, value=2.5, step=0.5)

        st.markdown("### 🌦️ Environmental & Soil Conditions")
        def_prof = crop_profiles[selected_crop]
        y_n = st.number_input("Nitrogen (N) kg/ha", value=int(def_prof['N'][0]))
        y_p = st.number_input("Phosphorus (P) kg/ha", value=int(def_prof['P'][0]))
        y_k = st.number_input("Potassium (K) kg/ha", value=int(def_prof['K'][0]))
        y_temp = st.number_input("Temperature (°C)", value=float(def_prof['temp'][0]))
        y_hum = st.number_input("Humidity (%)", value=float(def_prof['hum'][0]))
        y_ph = st.number_input("pH Level", value=float(def_prof['ph'][0]))
        y_rain = st.number_input("Rainfall (mm)", value=float(def_prof['rain'][0]))

        calc_yield = st.button("📊 Calculate Farm Production", use_container_width=True)

    with c2:
        st.markdown("### 💰 Harvest Projections")
        if calc_yield and predictor:
            y_res = predictor.predict_yield(
                crop=selected_crop, n=y_n, p=y_p, k=y_k,
                temp=y_temp, hum=y_hum, ph=y_ph, rain=y_rain,
                area_ha=area_ha
            )

            st.markdown(f"""
            <div class="yield-card">
                <h2 style="margin:0; color:#0d47a1;">{selected_crop.upper()} Production</h2>
                <h1 style="color:#1565c0; margin:10px 0;">{y_res['yield_tonnes_per_ha']} <span style="font-size:1.2rem;">Tonnes / Hectare</span></h1>
                <hr>
                <div style="font-size:1.1rem; margin-bottom:8px;"><b>Total Farm Output ({area_ha} Ha):</b></div>
                <div style="font-size:1.4rem; font-weight:bold; color:#0d47a1;">{y_res['total_production_tonnes']:.2f} Metric Tonnes</div>
                <div style="color:#333; margin-top:5px;">= <b>{y_res['total_production_quintals']:.1f}</b> Quintals | = <b>{y_res['total_production_kg']:,.0f}</b> Kilograms</div>
            </div>
            """, unsafe_allow_html=True)

            # Sensitivity Simulation (Rainfall impact)
            st.markdown("#### 🌧️ Rainfall Sensitivity Simulator")
            rain_variations = [0.7, 0.85, 1.0, 1.15, 1.3]
            sim_yields = []
            for r_factor in rain_variations:
                sim_r = y_rain * r_factor
                res_sim = predictor.predict_yield(
                    crop=selected_crop, n=y_n, p=y_p, k=y_k,
                    temp=y_temp, hum=y_hum, ph=y_ph, rain=sim_r,
                    area_ha=area_ha
                )
                sim_yields.append({
                    'Rainfall Scenario': f"{int((r_factor-1)*100):+d}% ({sim_r:.0f} mm)",
                    'Yield (Tonnes/Ha)': res_sim['yield_tonnes_per_ha'],
                    'Total Tonnes': res_sim['total_production_tonnes']
                })
            
            sim_df = pd.DataFrame(sim_yields)
            fig_sim = px.line(
                sim_df, x='Rainfall Scenario', y='Yield (Tonnes/Ha)',
                markers=True, title=f"Yield Response to Rainfall Variations for {selected_crop.title()}"
            )
            st.plotly_chart(fig_sim, use_container_width=True)

# ==========================================
# MODULE 4: ANALYTICS & MODEL BENCHMARKS
# ==========================================
elif app_mode == "📊 Analytics & Model Benchmarks":
    st.markdown('<div class="main-header">📊 Machine Learning Benchmarks & EDA Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comparative evaluation metrics, confusion matrices, and feature importances.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🌾 Crop Classification Models",
        "📈 Yield Regression Models",
        "🔍 Diagnostic Visualizations",
        "📋 Dataset Explorer"
    ])

    with tab1:
        st.markdown("### 🏆 Stage 1: Crop Classification Benchmark (22 Crops)")
        if predictor and predictor.crop_metadata:
            meta = predictor.crop_metadata
            comp_df = pd.DataFrame(meta.get('benchmark_comparison', []))
            st.dataframe(comp_df, use_container_width=True)

            best_m = meta.get('best_model')
            best_f1 = meta.get('best_metrics', {}).get('F1_Score', 0)
            st.success(f"Top Classifier: **{best_m}** with Weighted F1-Score of **{best_f1*100:.2f}%**")

            # Feature Importance
            if 'feature_importances' in meta:
                fi_df = pd.DataFrame(meta['feature_importances'])
                fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', title="Agronomic Feature Importance Ranking")
                st.plotly_chart(fig_fi, use_container_width=True)

    with tab2:
        st.markdown("### 🏆 Stage 2: Yield Regression Benchmark")
        if predictor and predictor.yield_metadata:
            y_meta = predictor.yield_metadata
            y_comp_df = pd.DataFrame(y_meta.get('benchmark_comparison', []))
            st.dataframe(y_comp_df, use_container_width=True)

            best_r = y_meta.get('best_model')
            best_r2 = y_meta.get('best_metrics', {}).get('R2_Score', 0)
            best_rmse = y_meta.get('best_metrics', {}).get('RMSE', 0)
            st.success(f"Top Regressor: **{best_r}** with R² of **{best_r2:.4f}** and RMSE of **{best_rmse:.4f} Tonnes/Ha**")

    with tab3:
        st.markdown("### 🖼️ Diagnostic Plots & Heatmaps")
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            if os.path.exists("visualizations/crop_confusion_matrix.png"):
                st.image("visualizations/crop_confusion_matrix.png", caption="Crop Classification Confusion Matrix", use_container_width=True)
            if os.path.exists("visualizations/crop_model_comparison.png"):
                st.image("visualizations/crop_model_comparison.png", caption="Classifier Performance Comparison", use_container_width=True)
        with v_col2:
            if os.path.exists("visualizations/yield_actual_vs_predicted.png"):
                st.image("visualizations/yield_actual_vs_predicted.png", caption="Yield Actual vs Predicted Parity Plot", use_container_width=True)
            if os.path.exists("visualizations/yield_model_comparison.png"):
                st.image("visualizations/yield_model_comparison.png", caption="Regressor Performance Comparison", use_container_width=True)

    with tab4:
        st.markdown("### 📁 Dataset Overview")
        if os.path.exists("data/Crop_recommendation.csv"):
            df_rec = pd.read_csv("data/Crop_recommendation.csv")
            st.markdown(f"**Crop Recommendation Dataset** (`{len(df_rec)}` rows, `{df_rec['label'].nunique()}` classes)")
            st.dataframe(df_rec.head(10), use_container_width=True)
            
            st.markdown("#### Numerical Feature Summary")
            st.dataframe(df_rec.describe(), use_container_width=True)

# ==========================================
# MODULE 5: ACADEMIC VIVA & ML THEORY HUB
# ==========================================
elif app_mode == "📚 Academic Viva & ML Theory Hub":
    st.markdown('<div class="main-header">📚 Academic Viva & Machine Learning Theory Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comprehensive theoretical foundations, mathematical formulations, and viva examination Q&A.</div>', unsafe_allow_html=True)

    st.markdown("### 🧮 Mathematical Formulations")
    c1, c2 = st.columns(2)
    with c1:
        with st.expander("🌲 Random Forest Ensemble Theory", expanded=True):
            st.markdown(r"""
            **Classification Formulation (Majority Voting):**
            $$\hat{C}(x) = \text{mode} \{ T_1(x), T_2(x), \dots, T_B(x) \}$$
            
            **Regression Formulation (Averaging):**
            $$\hat{y}(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$
            
            **Gini Impurity Metric:**
            $$I_G(p) = 1 - \sum_{i=1}^{J} p_i^2$$
            """)

        with st.expander("📊 Coefficient of Determination ($R^2$)", expanded=True):
            st.markdown(r"""
            $$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
            - Measures percentage of variance explained by model features.
            """)

    with c2:
        with st.expander("🧠 Gaussian Naive Bayes Formulation", expanded=True):
            st.markdown(r"""
            **Bayes' Rule with Conditional Independence:**
            $$P(y_k \mid x) \propto P(y_k) \prod_{i=1}^d \frac{1}{\sqrt{2\pi\sigma_{ik}^2}} \exp\left(-\frac{(x_i - \mu_{ik})^2}{2\sigma_{ik}^2}\right)$$
            """)

        with st.expander("🎯 Radial Basis Function (RBF) Kernel for SVM", expanded=True):
            st.markdown(r"""
            $$K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right)$$
            - Maps non-linear agronomic soil features into infinite-dimensional Hilbert space.
            """)

    st.markdown("---")
    st.markdown("### 🎓 Academic Viva Examination Questions & Answers")

    viva_qa = [
        ("Q1: Why is a two-stage decoupled architecture (Classification + Regression) better than a single model?",
         "Crop suitability recommendation is inherently a discrete multi-class categorical decision (which variety matches this soil and climate), whereas yield prediction is a continuous numeric estimation that depends on the specific crop variety and farm land area. Decoupling the pipeline allows dedicated optimization of both stages and provides transparent decision paths."),
        
        ("Q2: Why does Random Forest perform exceptionally well on soil macronutrient data?",
         "Soil nutrients (N, P, K) exhibit strong non-linear thresholds (e.g. crop yield plateaus or drops if Nitrogen is either deficient or toxic). Random Forest builds decorrelated decision trees using bagging and random feature subspaces, preventing overfitting while capturing sharp agronomic thresholds."),
         
        ("Q3: How does the pipeline handle categorical variables like Crop Name in the Yield Model?",
         "We use a Scikit-Learn `ColumnTransformer` with `OneHotEncoder(handle_unknown='ignore')` to convert nominal crop names into binary indicator vectors, alongside `StandardScaler` for continuous numeric features like rainfall, temperature, and area."),
         
        ("Q4: How does this project align with United Nations Sustainable Development Goals (SDGs)?",
         "The project aligns directly with **SDG 2 (Zero Hunger)** by optimizing crop productivity and preventing crop failure, and **SDG 12 (Responsible Consumption and Production)** by preventing excessive synthetic chemical fertilizer overuse through targeted soil-nutrient matching.")
    ]

    for q, a in viva_qa:
        with st.expander(f"📌 {q}"):
            st.markdown(f"> **Answer:** {a}")
