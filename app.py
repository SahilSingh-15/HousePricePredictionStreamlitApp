import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, features, and confidence dictionary
model = joblib.load("data/housing_model.pkl")
scaler = joblib.load("data/scaler.pkl")
features = joblib.load("data/model_features.pkl")
confidence_lookup = joblib.load("data/confidence_lookup.pkl")

st.title("🏠 Housing Price Prediction App")
st.write("This app predicts California housing prices based on input features.")

# User Input Form
with st.form("prediction_form"):
    st.subheader("Enter the details:")

    longitude = st.number_input("Longitude", value=-118.0)
    latitude = st.number_input("Latitude", value=34.0)
    housing_median_age = st.slider("Housing Median Age", 1, 100, 25)
    total_rooms = st.number_input("Total Rooms", value=2000)
    population = st.number_input("Population", value=1000)
    households = st.number_input("Households", value=500)
    median_income = st.number_input("Median Income (in 10,000s)", value=3.0)
    
    ocean_proximity = st.selectbox("Ocean Proximity", 
        ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

    # Margin of error
    margin = st.slider("Select Margin of Error (%)", 1, 50, 10)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Prepare input
    user_input = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }])

    user_encoded = pd.get_dummies(user_input)
    user_encoded = user_encoded.reindex(columns=features, fill_value=0)

    user_scaled = scaler.transform(user_encoded)
    prediction = model.predict(user_scaled)[0]

    # Calculate bounds
    margin_value = (margin / 100) * prediction
    lower = prediction - margin_value
    upper = prediction + margin_value

    # Confidence from precomputed dict
    confidence = confidence_lookup.get(margin, "N/A")

    st.success(f"🏡 Predicted Median House Value: ${prediction:,.2f}")
    st.info(f"📊 Estimated Range (±{margin}%): ${lower:,.2f} - ${upper:,.2f}")
    st.write(f"✅ Model Confidence within ±{margin}%: **{confidence:.2f}%**")
