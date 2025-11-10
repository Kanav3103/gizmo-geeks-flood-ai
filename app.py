# ==============================
# 🌊 Flood Prediction AI Dashboard (Simulated Formula-Based)
# ==============================
import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium

# =====================================
# 🔸 CONSTANTS: feature names and max values
# =====================================
FEATURES = [
    "MonsoonIntensity",
    "TopographyDrainage",
    "ClimateChange",
    "DamsQuality",
    "Siltation",
    "AgriculturalPractices",
    "DrainageSystems",
    "CoastalVulnerability",
    "Landslides",
    "Watersheds"
]

# Max values for each feature (as provided)
FEATURE_MAX = {
    "MonsoonIntensity": 16,
    "TopographyDrainage": 18,
    "ClimateChange": 17,
    "DamsQuality": 16,
    "Siltation": 16,
    "AgriculturalPractices": 16,
    "DrainageSystems": 17,
    "CoastalVulnerability": 17,
    "Landslides": 16,
    "Watersheds": 16
}

TARGET_MIN = 0.28
TARGET_MAX = 0.72

# =====================================
# 🔸 HELPER — map inputs into features
# =====================================
def map_user_inputs_to_features(rainfall, humidity, temperature, soil):
    mapped = {}
    mapped["MonsoonIntensity"] = rainfall / 10.0
    mapped["TopographyDrainage"] = max(0.0, 1.0 - soil / 120.0) * FEATURE_MAX["TopographyDrainage"]
    mapped["ClimateChange"] = (temperature / 45.0) * FEATURE_MAX["ClimateChange"]
    mapped["DamsQuality"] = max(0.0, 1.0 - rainfall / 180.0) * FEATURE_MAX["DamsQuality"]
    mapped["Siltation"] = ((rainfall + soil) / 200.0) * FEATURE_MAX["Siltation"]
    mapped["AgriculturalPractices"] = (soil / 100.0) * FEATURE_MAX["AgriculturalPractices"]
    mapped["DrainageSystems"] = max(0.0, 1.0 - soil / 110.0) * FEATURE_MAX["DrainageSystems"]
    mapped["CoastalVulnerability"] = ((humidity + (rainfall / 6.0)) / 110.0) * FEATURE_MAX["CoastalVulnerability"]
    mapped["Landslides"] = ((rainfall + soil) / 240.0) * FEATURE_MAX["Landslides"]
    mapped["Watersheds"] = max(0.0, 1.0 - rainfall / 300.0) * FEATURE_MAX["Watersheds"]

    for feat in FEATURES:
        val = float(mapped.get(feat, 0.0))
        val = max(0.0, min(val, FEATURE_MAX[feat]))
        mapped[feat] = val
    return mapped

# =====================================
# 🔸 FORMULA-BASED FLOOD RISK
# =====================================
def calculate_flood_probability(rainfall, humidity, temperature, soil):
    """Smart simulation of flood probability using realistic relationships."""
    rainfall = float(rainfall)
    humidity = float(humidity)
    temperature = float(temperature)
    soil = float(soil)

    # Derived metrics
    monsoon = rainfall / 15
    drainage = max(0, 1 - soil / 120)
    heat_factor = temperature / 45
    humidity_factor = humidity / 100
    soil_factor = soil / 100
    rainfall_factor = rainfall / 400

    flood_score = (
        0.30 * rainfall_factor +
        0.20 * humidity_factor +
        0.15 * heat_factor +
        0.20 * (1 - drainage) +
        0.15 * soil_factor
    )

    flood_score = np.clip(flood_score, 0, 1)
    return flood_score

# =====================================
# 🔸 SAFETY GUIDE
# =====================================
safety_guide = {
    (0, 10): {"Before": "Keep checking daily weather forecasts and stay updated. Clean drains and gutters around your home to ensure smooth water flow. Stay aware, even if flood chances seem low.",
              "During": "No major risk, but stay cautious if heavy rain continues. Avoid unnecessary travel during rainfall. Keep your emergency contacts handy just in case.",
              "After": "Inspect your surroundings for waterlogging or leaks. Dry out damp areas to prevent mosquito breeding. Continue monitoring local weather updates."},
    (10, 20): {"Before": "Monitor rainfall and river level trends closely. Prepare essential supplies like a torch, batteries, and first aid kit. Ensure your family knows basic emergency numbers.",
               "During": "Avoid walking in puddles or small flooded areas. Keep all electronics unplugged during lightning or storms. Monitor local alerts or advisories carefully.",
               "After": "Clean surroundings to prevent mosquito growth. Dispose of any waterlogged waste promptly. Be alert for early signs of disease or contamination."},
    (90, 100): {"Before": "Full-scale flooding possible — immediate preparation required. Evacuate low-lying zones early to avoid being trapped. Ensure pets, elderly, and children are moved first.",
                "During": "Call emergency helplines if trapped or isolated. Avoid rooftops unless it’s the only option and signal for help. Stay calm and conserve phone battery.",
                "After": "Wait for official clearance before re-entry. Thoroughly disinfect all water and food supplies. Assist community members in post-flood recovery."},
    # (You can keep all the other ranges — same as before)
}

# =====================================
# 🔸 STREAMLIT UI
# =====================================
st.set_page_config(page_title="Flood Prediction AI", layout="wide")
st.title("🌧️ Flood Prediction & Safety Dashboard")

tabs = st.tabs([
    "🌆 Mumbai Live Data",
    "🔍 Predict Flood Risk",
    "🛟 Flood Safety Guide",
    "🚨 Emergency Helplines",
    "🧭 Evacuation Route & Safe Shelters"
])

# ---------------- TAB 1 ----------------
with tabs[0]:
    st.header("🌆 Mumbai Live Data (Automatically updated from Satellites)")
    st.write("This data is simulated and can be replaced with live API data later.")

    mumbai_data = {
        "Rainfall (mm)": 215,
        "Humidity (%)": 82,
        "Temperature (°C)": 29,
        "Soil Moisture (%)": 55
    }

    df_mumbai = pd.DataFrame([mumbai_data])
    st.table(df_mumbai)

    # Formula-based risk
    flood_prob = calculate_flood_probability(
        rainfall=mumbai_data["Rainfall (mm)"],
        humidity=mumbai_data["Humidity (%)"],
        temperature=mumbai_data["Temperature (°C)"],
        soil=mumbai_data["Soil Moisture (%)"]
    )

    risk = round(flood_prob * 100, 2)
    st.subheader(f"Predicted Flood Risk: {risk}%")

    for (low, high), guide in safety_guide.items():
        if low <= risk <= high:
            st.markdown(f"### 🛟 Flood Safety Actions for Mumbai ({low}-{high}% Risk)")
            st.markdown(f"**Before Flood:** {guide['Before']}")
            st.markdown(f"**During Flood:** {guide['During']}")
            st.markdown(f"**After Flood:** {guide['After']}")
            break

# ---------------- TAB 2 ----------------
with tabs[1]:
    st.header("🔍 Predict Flood Risk Manually")
    rainfall = st.number_input("Rainfall (mm)", 0.0, 1000.0, 200.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0)
    temperature = st.number_input("Temperature (°C)", -10.0, 60.0, 28.0)
    soil = st.number_input("Soil Moisture (%)", 0.0, 100.0, 40.0)

    if st.button("Predict Risk"):
        flood_prob = calculate_flood_probability(rainfall, humidity, temperature, soil)
        risk_percent = round(flood_prob * 100, 2)
        st.subheader(f"Predicted Flood Risk: {risk_percent}%")

        with st.expander("Show mapped features"):
            mapped = map_user_inputs_to_features(rainfall, humidity, temperature, soil)
            feat_df = pd.DataFrame([{k: mapped[k] for k in FEATURES}])
            st.dataframe(feat_df.T.rename(columns={0: "Value"}))

        for (low, high), guide in safety_guide.items():
            if low <= risk_percent <= high:
                st.markdown(f"### 🛟 Flood Safety Actions ({low}-{high}% Risk)")
                st.markdown(f"**Before Flood:** {guide['Before']}")
                st.markdown(f"**During Flood:** {guide['During']}")
                st.markdown(f"**After Flood:** {guide['After']}")
                break

# ---------------- TAB 3 ----------------
with tabs[2]:
    st.header("🛟 Flood Safety Guide — Check by Risk %")
    st.write("Enter a risk percentage to see tailored safety actions.")
    user_risk = st.slider("Enter Flood Risk %", 0, 100, 30)
    for (low, high), guide in safety_guide.items():
        if low <= user_risk <= high:
            st.markdown(f"### For {low}-{high}% Flood Risk:")
            st.markdown(f"**Before Flood:** {guide['Before']}")
            st.markdown(f"**During Flood:** {guide['During']}")
            st.markdown(f"**After Flood:** {guide['After']}")
            break

# ---------------- TAB 4 ----------------
with tabs[3]:
    st.header("🚨 Emergency Helplines & Disaster Contacts")
    st.markdown("""
    - **NDMA:** 011-26701700  
    - **Emergency (India):** 112  
    - **Disaster Control:** 1078  
    - **Ambulance:** 102 / 108  
    - **BMC Control Room:** 1916  
    - **Mumbai Police:** 100  
    - **Mumbai Flood Helpline:** 1916  
    """)

# ---------------- TAB 5 ----------------
with tabs[4]:
    st.header("🧭 Evacuation Route & Safe Shelters")
    evacuation_data = {
        "Andheri": {
            "center": [19.1197, 72.8468],
            "shelters": [
                {"name": "Andheri East Relief Camp", "lat": 19.1135, "lon": 72.8697},
                {"name": "Andheri Sports Complex Shelter", "lat": 19.1260, "lon": 72.8360}
            ],
            "route": [[19.1135, 72.8697], [19.1197, 72.8468], [19.1260, 72.8360]]
        }
    }

    area = st.selectbox("Select the area closest to you:", ["Andheri"])
    data = evacuation_data[area]

    m = folium.Map(location=data["center"], zoom_start=13)
    for shelter in data["shelters"]:
        folium.Marker([shelter["lat"], shelter["lon"]], popup=shelter["name"], icon=folium.Icon(color="green")).add_to(m)

    folium.PolyLine(data["route"], color="blue", weight=3, opacity=0.7).add_to(m)
    st_folium(m, width=700, height=500)

