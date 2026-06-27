import streamlit as st
import pandas as pd
import joblib
import os
import requests

st.set_page_config(page_title="AQI predictor", page_icon="🌫️", layout="centered")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aqi_model.joblib")

@st.cache_resource
def load_bundle():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}. Files in this directory: {os.listdir(os.path.dirname(MODEL_PATH))}")
        st.stop()
    return joblib.load(MODEL_PATH)

bundle = load_bundle()
model = bundle["model"]
le = bundle["label_encoder"]
feature_cols = bundle["feature_cols"]
cities = bundle["cities"]
ranges = bundle["ranges"]

CITY_COORDS = {
    "Ahmedabad": (23.0225, 72.5714), "Aizawl": (23.7271, 92.7176), "Amaravati": (16.5062, 80.6480),
    "Amritsar": (31.6340, 74.8723), "Bengaluru": (12.9716, 77.5946), "Bhopal": (23.2599, 77.4126),
    "Brajrajnagar": (21.8167, 83.9167), "Chandigarh": (30.7333, 76.7794), "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558), "Delhi": (28.7041, 77.1025), "Ernakulam": (9.9816, 76.2999),
    "Gurugram": (28.4595, 77.0266), "Guwahati": (26.1445, 91.7362), "Hyderabad": (17.3850, 78.4867),
    "Jaipur": (26.9124, 75.7873), "Jorapokhar": (23.7090, 86.4140), "Kochi": (9.9312, 76.2673),
    "Kolkata": (22.5726, 88.3639), "Lucknow": (26.8467, 80.9462), "Mumbai": (19.0760, 72.8777),
    "Patna": (25.5941, 85.1376), "Shillong": (25.5788, 91.8933), "Talcher": (20.9500, 85.2167),
    "Thiruvananthapuram": (8.5241, 76.9366), "Visakhapatnam": (17.6868, 83.2185),
}

st.title("AQI predictor")
st.caption("Random forest model trained on the City Day air quality dataset.")

city = st.selectbox("City", cities, index=cities.index("Delhi") if "Delhi" in cities else 0)

if "fetched" not in st.session_state:
    st.session_state.fetched = {}

fetch_col, status_col = st.columns([1, 3])
if fetch_col.button("Fetch live readings"):
    api_key = st.secrets.get("OPENWEATHER_API_KEY", "")
    if not api_key:
        status_col.error("No API key configured. Add OPENWEATHER_API_KEY in Streamlit secrets.")
    elif city not in CITY_COORDS:
        status_col.warning(f"No coordinates on file for {city}.")
    else:
        lat, lon = CITY_COORDS[city]
        try:
            resp = requests.get(
                "https://api.openweathermap.org/data/2.5/air_pollution",
                params={"lat": lat, "lon": lon, "appid": api_key},
                timeout=10,
            )
            resp.raise_for_status()
            comp = resp.json()["list"][0]["components"]
            st.session_state.fetched = {
                "PM2.5": comp.get("pm2_5"),
                "NO": comp.get("no"),
                "NO2": comp.get("no2"),
                "SO2": comp.get("so2"),
                "O3": comp.get("o3"),
                "CO": comp.get("co", 0) / 1000,
            }
            status_col.success(f"Pulled live readings for {city}.")
        except Exception as e:
            status_col.error(f"Could not fetch live data: {e}")

live = st.session_state.fetched

st.subheader("Pollutant levels")
st.caption("PM2.5, NO, NO2, SO2, O3 and CO can be auto-filled from live data. NOx, Benzene and Toluene aren't available from public air-quality APIs, so set those manually.")

col1, col2 = st.columns(2)

def slider_for(label, key, col, unit_scale=1.0):
    lo, hi = ranges[key]
    default = live.get(key)
    if default is None:
        default = (lo + hi) / 2
    default = max(lo, min(hi, default * unit_scale))
    return col.slider(label, min_value=round(lo, 1), max_value=round(hi, 1), value=round(default, 1))

pm25 = slider_for("PM2.5 (μg/m³)", "PM2.5", col1)
no = slider_for("NO (μg/m³)", "NO", col1)
no2 = slider_for("NO2 (μg/m³)", "NO2", col1)
nox = slider_for("NOx (μg/m³)", "NOx", col1)
co = slider_for("CO (mg/m³)", "CO", col1)
so2 = slider_for("SO2 (μg/m³)", "SO2", col2)
o3 = slider_for("O3 (μg/m³)", "O3", col2)
benzene = slider_for("Benzene (μg/m³)", "Benzene", col2)
toluene = slider_for("Toluene (μg/m³)", "Toluene", col2)

city_encoded = le.transform([city])[0]

input_df = pd.DataFrame([{
    "City_encoded": city_encoded,
    "PM2.5": pm25,
    "NO": no,
    "NO2": no2,
    "NOx": nox,
    "CO": co,
    "SO2": so2,
    "O3": o3,
    "Benzene": benzene,
    "Toluene": toluene,
}])[feature_cols]

predicted_aqi = model.predict(input_df)[0]

def get_mitigation_measures(aqi):
    if aqi <= 50:
        return "Good", [], []
    elif aqi <= 100:
        return "Satisfactory", ["Sensitive groups should be cautious"], []
    elif aqi <= 150:
        return "Unhealthy for sensitive groups", ["Sensitive groups avoid outdoor activity"], []
    elif aqi <= 200:
        return "Unhealthy", ["Everyone should reduce outdoor activity"], ["Wear masks", "Use air purifiers"]
    elif aqi <= 300:
        return "Very unhealthy", ["Avoid outdoor activity"], ["Stay indoors", "Use masks"]
    else:
        return "Hazardous", ["Stay indoors", "Avoid all outdoor activity"], ["Emergency measures"]

category, warnings, actions = get_mitigation_measures(predicted_aqi)

st.divider()
st.metric("Predicted AQI", f"{predicted_aqi:.1f}", category)

if predicted_aqi > 150:
    st.error(f"Alert: {category}. " + " | ".join(warnings))
    if actions:
        st.write("Recommended actions: " + ", ".join(actions))
else:
    st.success(f"No alert. Category: {category}")
