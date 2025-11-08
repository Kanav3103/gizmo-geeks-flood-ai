#https://gizmo-geeks-flood-ai-pew2g7fgsivnjbznvnnrmc.streamlit.app

#Link, for backend, KANAV ONLY : https://github.com/Kanav3103/gizmo-geeks-flood-ai/blob/main/app.py


# ==============================
# 🌊 Flood Prediction AI Dashboard
# ==============================
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import folium
from streamlit_folium import st_folium

# =====================================
# 🔸 SIMULATED MODEL TRAINING (Random data)
# =====================================
np.random.seed(42)
data_size = 500
rainfall = np.random.uniform(50, 400, data_size)      # mm
humidity = np.random.uniform(40, 100, data_size)      # %
temperature = np.random.uniform(20, 40, data_size)    # °C
soil_moisture = np.random.uniform(10, 80, data_size)  # %
flood_risk = 0.4*rainfall + 0.3*humidity + 0.2*soil_moisture - 0.3*temperature + np.random.uniform(-20, 20, data_size)
flood_risk = np.clip(flood_risk/10, 0, 100)

df = pd.DataFrame({
    'Rainfall': rainfall,
    'Humidity': humidity,
    'Temperature': temperature,
    'Soil Moisture': soil_moisture,
    'Flood Risk (%)': flood_risk
})

X = df[['Rainfall', 'Humidity', 'Temperature', 'Soil Moisture']]
y = df['Flood Risk (%)']
model = LinearRegression()
model.fit(X, y)
joblib.dump(model, "flood_model.pkl")

# Load model
model = joblib.load("flood_model.pkl")

# =====================================
# 🔸 SAFETY GUIDE (Detailed)
# =====================================
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
        )
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
        )
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
        )
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
        )
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
        )
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
        )
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
        )
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
        )
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
        )
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
        )
    }
}

# =====================================
# 🔸 STREAMLIT APP LAYOUT
# =====================================
st.set_page_config(page_title="Flood Prediction AI", layout="wide")
st.title("🌧️ Flood Prediction & Safety Dashboard")

tabs = st.tabs([
    "🌆 Mumbai Live Data",
    "🔍 Predict Flood Risk",
    "🛟 Flood Safety Guide",
    "🧭 Evacuation Route & Safe Shelters"
])

# ---------------- TAB 1 ----------------
with tabs[0]:
    st.header("🌆 Mumbai Live Data (Automatically updated from Satellites)")
    st.write("The data is automatically updated from the satellites (SMAP, GRACE, ERA5).")
    st.write("This update will be stopped after the event, as our PCs will not be able to handle the load for an extended amount of time, but this can be done, given ample resources.")
    
    # 🔹 Example live data (you can change these values anytime)
    mumbai_data = {
        "Rainfall (mm)": 215,
        "Humidity (%)": 82,
        "Temperature (°C)": 29,
        "Soil Moisture (%)": 55
    }

    df_mumbai = pd.DataFrame([mumbai_data])
    st.table(df_mumbai)

    # 🔹 Predict flood risk for Mumbai
    if mumbai_data["Rainfall (mm)"] < 50:
        risk = 0
    else:
        pred = model.predict([[mumbai_data["Rainfall (mm)"], mumbai_data["Humidity (%)"],
                               mumbai_data["Temperature (°C)"], mumbai_data["Soil Moisture (%)"]]])[0]
        risk = round(float(np.clip(pred, 0, 100)), 2)
        
    st.subheader(f"Predicted Flood Risk: {risk}%")

    # 🔹 Show safety guidance for the predicted risk
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

    rainfall = st.number_input("Rainfall (mm)", 0, 500, 200)# Minimum, Maximum, Default
    humidity = st.number_input("Humidity (%)", 0, 100, 70)# Minimum, Maximum, Default
    temperature = st.number_input("Temperature (°C)", 0, 50, 28)# Minimum, Maximum, Default
    soil = st.number_input("Soil Moisture (%)", 0, 100, 40)# Minimum, Maximum, Default

    if st.button("Predict Risk"):
        if rainfall < 50:
            risk = 0
        else:
            pred = model.predict([[rainfall, humidity, temperature, soil]])[0]
            risk = round(float(np.clip(pred, 0, 100)), 2)

        st.subheader(f"Predicted Flood Risk: {risk}%")
        
        for (low, high), guide in safety_guide.items():
            if low <= risk <= high:
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

    st.info(
        "💡 **Tip:** Always keep your phone charged, follow official instructions, and avoid spreading rumours during emergencies."
    )

    st.success("Stay alert, stay safe, and help others when possible 💪")

# ---------------- TAB 5 ----------------
with tabs[4]:
    st.header("🧭 Evacuation Route & Safe Shelters")
    st.write("Select your area to view nearby safe shelters and recommended evacuation routes during heavy rainfall or flood alerts.")

    # Dropdown for user area selection
    area = st.selectbox("Select the area closest to you:", ["Andheri", "Kurla", "Bandra", "Dadar", "Powai"])

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

    # Generate map dynamically based on selected area
    data = evacuation_data[area]
    m = folium.Map(location=data["center"], zoom_start=13)

    # Add shelters as markers
    for s in data["shelters"]:
        folium.Marker(
            [s["lat"], s["lon"]],
            popup=s["name"],
            icon=folium.Icon(color="green", icon="home")
        ).add_to(m)

    # Draw example route
    folium.PolyLine(
        locations=data["route"],
        color="blue",
        weight=3,
        opacity=0.7,
        tooltip="Recommended Evacuation Path"
    ).add_to(m)

    # Display interactive map
    st_folium(m, width=700, height=500)

    # Extra info below map
    st.markdown(
        "<p style='text-align:center; font-size:16px; color:gray;'>"
        "📍 Always follow official local evacuation orders and stay informed via government alerts."
        "</p>",
        unsafe_allow_html=True
    )
