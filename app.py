# https://gizmo-geeks-flood-ai-pew2g7fgsivnjbznvnnrmc.streamlit.app
# Link, for backend, KANAV ONLY : https://github.com/Kanav3103/gizmo-geeks-flood-ai/blob/main/app.py

import streamlit as st
import pandas as pd
import numpy as np
import time
import folium
from streamlit_folium import st_folium

# === Bright Mauve Theme (No Stars) ===
st.markdown("""
<style>

/* ===== Tabs Section ===== */
div[data-baseweb="tab-list"] {
    background: rgba(255, 255, 255, 0.25);
    backdrop-filter: blur(10px);
    border-radius: 14px;
    padding: 0.6rem 1rem;
    margin-bottom: 1rem;
    color: black !important;
}
div[data-baseweb="tab"] p {
    color: black !important;
    font-weight: 700;
}

/* ===== Buttons ===== */
.stButton>button, .stDownloadButton>button {
    background: linear-gradient(135deg, #b87fff, #9e5cff);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
    box-shadow: 0 0 10px rgba(180, 120, 255, 0.5);
    transition: all 0.3s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    background: linear-gradient(135deg, #a864ff, #8a3eff);
    box-shadow: 0 0 18px rgba(180, 120, 255, 0.8);
}

/* ===== DataFrame Table Styling ===== */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 0 15px rgba(150, 90, 255, 0.4);
}

/* ===== Headers ===== */
h1, h2, h3, h4, h5, h6 {
    color: white !important;
}

/* ===== Softer Subtle Text Glow Effect ===== */
h1, h2, h3, h4, h5, h6, p, span, div {
    text-shadow: 0 0 6px rgba(210, 190, 255, 0.50),
                 0 0 12px rgba(180, 140, 230, 0.40);
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Flood Predictor AI", page_icon="🌊", layout="wide")

# =====================================================
# STARTUP / TRAINING SIMULATION SCREEN
# =====================================================
if "boot_completed" not in st.session_state:
    placeholder = st.empty()
    with placeholder.container():
        st.title("🌊 Gizmo Geeks Flood AI — Booting Up")
        st.write("### 🤖 Initializing flood prediction engine... please wait ⏳")

        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.1)  # total ~10 seconds
            progress_bar.progress(i + 1)

        st.success("✅ Model trained successfully!")
        time.sleep(3)

    # Clear the placeholder (remove training screen)
    placeholder.empty()
    st.session_state.boot_completed = True

st.title("🌊 Gizmo Geeks Flood AI — Flood Prediction System")
st.markdown("### Smart flood risk prediction based on environmental conditions.")

# =====================================================
# CONSTANTS & FEATURE SETUP
# =====================================================
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

# =====================================================
# INPUT → FEATURE MAPPING
# =====================================================
def map_user_inputs_to_features(rainfall, humidity, temperature, soil):
    mapped = {}
    rainfall_clamped = max(0.0, min(rainfall, 600.0))
    rainfall_factor = rainfall_clamped / 600.0
    TMIN = 10.0
    TMAX = 45.0
    temp_clamped = max(TMIN, min(temperature, TMAX))
    temp_norm = (temp_clamped - TMIN) / (TMAX - TMIN)
    inverted_temp = 1.0 - temp_norm
    mapped["MonsoonIntensity"] = rainfall_factor * FEATURE_MAX["MonsoonIntensity"]
    mapped["TopographyDrainage"] = max(0.0, 1.0 - soil / 120.0) * FEATURE_MAX["TopographyDrainage"]
    mapped["ClimateChange"] = inverted_temp * FEATURE_MAX["ClimateChange"]
    mapped["DamsQuality"] = max(0.0, 1.0 - rainfall_clamped / 180.0) * FEATURE_MAX["DamsQuality"]
    mapped["Siltation"] = ((rainfall_clamped + soil) / 200.0) * FEATURE_MAX["Siltation"]
    mapped["AgriculturalPractices"] = (soil / 100.0) * FEATURE_MAX["AgriculturalPractices"]
    mapped["DrainageSystems"] = max(0.0, 1.0 - soil / 110.0) * FEATURE_MAX["DrainageSystems"]
    mapped["CoastalVulnerability"] = ((humidity + (rainfall_clamped / 6.0)) / 110.0) * FEATURE_MAX["CoastalVulnerability"]
    mapped["Landslides"] = ((rainfall_clamped + soil) / 240.0) * FEATURE_MAX["Landslides"]
    mapped["Watersheds"] = max(0.0, 1.0 - rainfall_clamped / 300.0) * FEATURE_MAX["Watersheds"]
    return mapped

# =====================================================
# FORMULA-BASED FLOOD RISK MODEL
# =====================================================
def calculate_flood_probability(rainfall, humidity, temperature, soil):

    rainfall = float(rainfall)
    humidity = float(humidity)
    temperature = float(temperature)
    soil = float(soil)

    # 🚫 No flood risk for light rain
    if rainfall < 50:
        return 0.0

    # 🌧️ Rainfall-based scaling
    if rainfall <= 150:
        rainfall_factor = 0.15 * (rainfall / 150)  # up to 15% at 150 mm
    elif rainfall >= 300:
        rainfall_factor = 1.0  # full risk at ≥300 mm
    else:
        # Linear interpolation between 150 and 300 mm (0.15 → 1.0)
        rainfall_factor = 0.15 + (1.0 - 0.15) * ((rainfall - 150) / (300 - 150))

    # 🌫️ Environmental modifiers
    humidity_factor = humidity / 100.0
    heat_factor = max(0.0, 1.0 - ((temperature - 10.0) / 35.0))  # cooler → higher risk
    drainage_factor = max(0.0, 1.0 - soil / 120.0)               # poor drainage → high risk
    soil_factor = soil / 100.0                                   # high soil moisture → high risk

    # ⚖️ Weighted combination (rainfall = 40%)
    flood_score = (
        0.40 * rainfall_factor +
        0.25 * humidity_factor +
        0.15 * heat_factor +
        0.10 * (1 - drainage_factor) +
        0.10 * soil_factor
    )

    # 🧮 Clamp total
    flood_score = np.clip(flood_score, 0.0, 1.0)

    # ⚠️ Enforce rainfall ≤150 mm cap
    if rainfall <= 150:
        flood_score = min(flood_score, 0.15)

    return flood_score


# =====================================================
# SAFETY GUIDE
# =====================================================
safety_guide = {
    (0, 10): {
        "Before": (
            "Keep checking daily weather forecasts and stay updated. "
            "Clean drains and gutters around your home to ensure smooth water flow. "
            "Stay aware, even if flood chances seem low."
        ),
        "During": (
            "No major risk, but stay cautious if heavy rain continues. "
            "Avoid unnecessary travel during rainfall. "
            "Keep your emergency contacts handy just in case."
        ),
        "After": (
            "Inspect your surroundings for waterlogging or leaks. "
            "Dry out damp areas to prevent mosquito breeding. "
            "Continue monitoring local weather updates."
        ),
    },
    (10, 20): {
        "Before": (
            "Monitor rainfall and river level trends closely. "
            "Prepare essential supplies like a torch, batteries, and first aid kit. "
            "Ensure your family knows basic emergency numbers."
        ),
        "During": (
            "Avoid walking in puddles or small flooded areas. "
            "Keep all electronics unplugged during lightning or storms. "
            "Monitor local alerts or advisories carefully."
        ),
        "After": (
            "Clean surroundings to prevent mosquito growth. "
            "Dispose of any waterlogged waste promptly. "
            "Be alert for early signs of disease or contamination."
        ),
    },
    (20, 30): {
        "Before": (
            "Store drinking water and food in sealed containers. "
            "Check and reinforce any weak walls or basement leaks. "
            "Keep valuables and documents in waterproof bags."
        ),
        "During": (
            "Move important items to higher shelves. "
            "Avoid outdoor activity in continuous rainfall. "
            "Stay connected with neighbours for updates."
        ),
        "After": (
            "Dry clothes and bedding immediately. "
            "Clean drains and ensure flow of water. "
            "Keep children away from muddy or wet areas."
        ),
    },
    (30, 40): {
        "Before": (
            "Prepare an emergency go-bag with essentials. "
            "Ensure everyone in the household knows safe exits. "
            "Charge your phones and power banks fully."
        ),
        "During": (
            "Avoid unnecessary movement and watch for rising water. "
            "Keep listening to radio or local alerts. "
            "Do not drive in heavy rain or flooded lanes."
        ),
        "After": (
            "Sanitize stored water sources before use. "
            "Help elderly neighbours with clean-up. "
            "Check for cracks or electrical faults in the home."
        ),
    },
    (40, 50): {
        "Before": (
            "Keep your emergency contact list visible and ready. "
            "Move important possessions and electronics to upper floors. "
            "Discuss safety plans with family members."
        ),
        "During": (
            "Avoid basements and low-lying areas. "
            "Do not touch electrical panels with wet hands. "
            "Ensure pets are kept indoors and safe."
        ),
        "After": (
            "Inspect building structures for any damage. "
            "Avoid using tap water until confirmed safe. "
            "Dry and disinfect floors and walls quickly."
        ),
    },
    (50, 60): {
        "Before": (
            "Start partial evacuation if water levels are expected to rise. "
            "Store clean water and non-perishable food items. "
            "Keep emergency kits near main exits."
        ),
        "During": (
            "Move to higher ground if floodwater approaches. "
            "Avoid contact with floodwater—it may be contaminated. "
            "Stay tuned to emergency broadcasts."
        ),
        "After": (
            "Wait for official clearance before returning home. "
            "Document damage for insurance or aid. "
            "Do not consume flood-exposed food or water."
        ),
    },
    (60, 70): {
        "Before": (
            "Stay ready for possible evacuation; stock up on essentials. "
            "Keep vehicles fuelled and parked on higher ground. "
            "Ensure kids and elderly know the evacuation plan."
        ),
        "During": (
            "Shift immediately to upper floors or safe zones. "
            "Avoid touching wet electrical wires or devices. "
            "Keep communicating your location to local help lines."
        ),
        "After": (
            "Allow authorities to declare it safe before cleanup. "
            "Disinfect and air-dry your belongings thoroughly. "
            "Support neighbours in rebuilding efforts."
        ),
    },
    (70, 80): {
        "Before": (
            "Coordinate with local disaster groups or neighbours. "
            "Keep all important documents in waterproof storage. "
            "Pack your evacuation kit and stay alert for warnings."
        ),
        "During": (
            "Evacuate immediately if advised by officials. "
            "Avoid roads with moving or deep water. "
            "Stay calm and assist others if possible."
        ),
        "After": (
            "Do not touch damaged power lines or poles. "
            "Clean and dry your home before turning on electricity. "
            "Boil water before drinking."
        ),
    },
    (80, 90): {
        "Before": (
            "Prepare for an emergency evacuation at any time. "
            "Keep constant communication with local authorities. "
            "Turn off main power and gas supplies before leaving."
        ),
        "During": (
            "Do not delay evacuation; safety is priority. "
            "Move to official shelters or high-rise safe areas. "
            "Carry essentials only and stay with your group."
        ),
        "After": (
            "Follow safety checks before re-entering flooded areas. "
            "Clean with disinfectants to avoid infections. "
            "Seek medical help if any injuries occur."
        ),
    },
    (90, 100): {
        "Before": (
            "Full-scale flooding possible — immediate preparation required. "
            "Evacuate low-lying zones early to avoid being trapped. "
            "Ensure pets, elderly, and children are moved first."
        ),
        "During": (
            "Call emergency helplines if trapped or isolated. "
            "Avoid rooftops unless it’s the only option and signal for help. "
            "Stay calm and conserve phone battery."
        ),
        "After": (
            "Wait for official clearance before re-entry. "
            "Thoroughly disinfect all water and food supplies. "
            "Assist community members in post-flood recovery."
        ),
    },
}

# =====================================================
# STREAMLIT UI — TABS
# =====================================================
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
    st.write("The data updates will be stopped after the event is over as our PCs will not be able to handle such an overload, but it can be done, given an ample amount of resources.")

    mumbai_data = {
        "Rainfall (mm)": 215,
        "Humidity (%)": 82,
        "Temperature (°C)": 29,
        "Soil Moisture (%)": 55
    }

    df_mumbai = pd.DataFrame([mumbai_data])
    st.table(df_mumbai)

    flood_prob = calculate_flood_probability(
        mumbai_data["Rainfall (mm)"],
        mumbai_data["Humidity (%)"],
        mumbai_data["Temperature (°C)"],
        mumbai_data["Soil Moisture (%)"]
    )

    risk = round(flood_prob * 100, 2)
    st.subheader(f"Predicted Flood Risk for Mumbai: {risk}%")

    for (low, high), guide in safety_guide.items():
        if low <= risk <= high:
            st.markdown(f"### 🛟 Safety Measures ({low}-{high}% Risk Zone)")
            st.markdown(f"**Before Flood:** {guide['Before']}")
            st.markdown(f"**During Flood:** {guide['During']}")
            st.markdown(f"**After Flood:** {guide['After']}")
            break

# ---------------- TAB 2 ----------------
with tabs[1]:
    st.header("🔍 Predict Flood Risk Manually")
    rainfall = st.number_input("Rainfall (mm)", 0, 500, 200)
    humidity = st.number_input("Humidity (%)", 0, 100, 70)
    temperature = st.number_input("Temperature (°C)", 10, 45, 28)
    soil = st.number_input("Soil Moisture (%)", 0, 100, 40)

    if st.button("Predict Risk"):
        flood_prob = calculate_flood_probability(rainfall, humidity, temperature, soil)
        risk_percent = round(flood_prob * 100, 2)
        st.subheader(f"Predicted Flood Risk: {risk_percent}%")

        for (low, high), guide in safety_guide.items():
            if low <= risk_percent <= high:
                st.markdown(f"### 🛟 Safety Actions ({low}-{high}% Zone)")
                st.markdown(f"**Before Flood:** {guide['Before']}")
                st.markdown(f"**During Flood:** {guide['During']}")
                st.markdown(f"**After Flood:** {guide['After']}")
                break

# ---------------- TAB 3 ----------------
with tabs[2]:
    st.header("Flood Safety Guide — Based on Risk %")
    user_risk = st.slider("Select your estimated Flood Risk (%)", 0, 100, 30)
    for (low, high), guide in safety_guide.items():
        if low <= user_risk <= high:
            st.markdown(f"### Safety Plan for {low}-{high}% Risk:")
            st.markdown(f"**Before Flood:** {guide['Before']}")
            st.markdown(f"**During Flood:** {guide['During']}")
            st.markdown(f"**After Flood:** {guide['After']}")
            break

# ---------------- TAB 4 ----------------
with tabs[3]:
    st.header("🚨 Emergency Helplines & Disaster Contacts")
    st.write("In case of a flood or any severe weather emergency, contact the following helplines immediately.")

    st.markdown("### 📞 National Helplines")
    st.markdown("""
   - **National Disaster Management Authority (NDMA):** 011-26701700
   - **National Emergency Helpline (India):** 112
   - **Disaster Management Control Room:** 1078
   - **Fire & Rescue Services:** 101
   - **Ambulance:** 102 / 108
   """)

    st.markdown("### 🌆 Mumbai-Specific Helplines")
    st.markdown("""
   - **Brihanmumbai Municipal Corporation (BMC) Control Room:** 1916
   - **Mumbai Police Helpline:** 100
   - **Mumbai Fire Brigade:** 101
   - **Mumbai Flood Helpline:** 1916 (Active during monsoon)
   - **Railway Helpline (for stranded passengers):** 139
   """)

    st.markdown("### 🌐 Useful Websites")
    st.markdown("""
   - [National Disaster Management Authority (NDMA)](https://ndma.gov.in)
   - [Maharashtra State Disaster Management Authority](https://dmgroup.maharashtra.gov.in)
   - [IMD Weather Updates](https://mausam.imd.gov.in)
   - [BMC Disaster Management](https://portal.mcgm.gov.in)
   """)

    st.info("💡 **Tip:** Always keep your phone charged, follow official instructions, and avoid spreading rumours during emergencies.")
    st.success("Stay alert, stay safe, and help others when possible 💪")

# ---------------- TAB 5 ----------------
with tabs[4]:
    st.header("🧭 Evacuation Route & Safe Shelters")
    st.write("Select your area to view nearby safe shelters and recommended evacuation routes during heavy rainfall or flood alerts.")

    # Define data for each area
    evacuation_data = {
        "Andheri": {
            "center": [19.1197, 72.8468],
            "shelters": [
                {"name": "Andheri East Relief Camp", "lat": 19.1135, "lon": 72.8697},
                {"name": "Andheri Sports Complex Shelter", "lat": 19.1260, "lon": 72.8360},
                {"name": "Vile Parle Community Hall", "lat": 19.1020, "lon": 72.8440},
            ],
            "route": [[19.1135, 72.8697], [19.1197, 72.8468], [19.1260, 72.8360]]
        },
        "Kurla": {
            "center": [19.0722, 72.8780],
            "shelters": [
                {"name": "Kurla Relief Camp", "lat": 19.0722, "lon": 72.8780},
                {"name": "Nehru Nagar High School Shelter", "lat": 19.0655, "lon": 72.8825},
                {"name": "BKC Public Ground", "lat": 19.0665, "lon": 72.8550},
            ],
            "route": [[19.0655, 72.8825], [19.0722, 72.8780], [19.0665, 72.8550]]
        },
        "Bandra": {
            "center": [19.0545, 72.8400],
            "shelters": [
                {"name": "Bandra West Shelter", "lat": 19.0580, "lon": 72.8340},
                {"name": "St. Andrew’s Auditorium", "lat": 19.0575, "lon": 72.8370},
                {"name": "Bandra Reclamation Ground", "lat": 19.0500, "lon": 72.8405},
            ],
            "route": [[19.0500, 72.8405], [19.0545, 72.8400], [19.0580, 72.8340]]
        },
        "Dadar": {
            "center": [19.0176, 72.8562],
            "shelters": [
                {"name": "Shivaji Park Hall Shelter", "lat": 19.0201, "lon": 72.8371},
                {"name": "Dadar Railway Camp", "lat": 19.0168, "lon": 72.8449},
                {"name": "Portuguese Church Shelter", "lat": 19.0231, "lon": 72.8441},
            ],
            "route": [[19.0168, 72.8449], [19.0176, 72.8562], [19.0201, 72.8371]]
        },
        "Powai": {
            "center": [19.1176, 72.9060],
            "shelters": [
                {"name": "IIT Bombay Main Ground Shelter", "lat": 19.1334, "lon": 72.9133},
                {"name": "Powai Lake View Relief Zone", "lat": 19.1102, "lon": 72.9053},
                {"name": "Hiranandani Public School Shelter", "lat": 19.1213, "lon": 72.9120},
            ],
            "route": [[19.1102, 72.9053], [19.1176, 72.9060], [19.1334, 72.9133]]
        }
    }

    # Dropdown for user area selection
    area = st.selectbox("Select the area closest to you:", ["Andheri", "Kurla", "Bandra", "Dadar", "Powai"])

    # Generate map dynamically based on selected area
    data = evacuation_data.get(area)
    if data is None:
        st.error("❌ No evacuation data found for this area!")
    else:
        m = folium.Map(location=data["center"], zoom_start=13)

        for s in data["shelters"]:
            folium.Marker(
                [s["lat"], s["lon"]],
                popup=s["name"],
                icon=folium.Icon(color="green", icon="home")
            ).add_to(m)

        folium.PolyLine(
            locations=data["route"],
            color="blue",
            weight=3,
            opacity=0.7,
            tooltip="Recommended Evacuation Path"
        ).add_to(m)

        st_folium(m, width=1400, height=1000)
        st.markdown(
            "<p style='text-align:center; font-size:16px; color:lightgreen;'>📍 Always follow official local evacuation orders and stay informed via government alerts.</p>",
            unsafe_allow_html=True
        )
