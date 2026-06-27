import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="AQI predictor", page_icon="🌫️", layout="centered")

@st.cache_resource
def load_bundle():
    return joblib.load("aqi_model.joblib")

bundle = load_bundle()
model = bundle["model"]
le = bundle["label_encoder"]
feature_cols = bundle["feature_cols"]
cities = bundle["cities"]
ranges = bundle["ranges"]

st.title("AQI predictor")
st.caption("Random forest model trained on the City Day air quality dataset.")

city = st.selectbox("City", cities, index=cities.index("Delhi") if "Delhi" in cities else 0)

st.subheader("Pollutant levels")

col1, col2 = st.columns(2)

def slider_for(label, key, col):
    lo, hi = ranges[key]
    default = (lo + hi) / 2
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
