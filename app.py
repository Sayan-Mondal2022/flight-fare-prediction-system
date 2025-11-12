# app.py
import streamlit as st
import pandas as pd
import joblib
from model_preprocess import preprocess_custom_input

# Load trained model and feature columns
model = joblib.load("model.pkl")
feature_columns = joblib.load("feature_columns.pkl")  # Saved during training

# Streamlit Page Config
st.set_page_config(page_title="✈️ Flight Fare Prediction", layout="centered")

# Title and Description
st.title("✈️ Flight Fare Prediction System")
st.markdown("Predict flight ticket prices based on your preferences and schedule.")

# Input Form
with st.form("fare_form"):
    col1, col2 = st.columns(2)

    with col1:
        total_stops = st.selectbox("Total Stops", ["non-stop", "1-stop", "2+-stop"])
        travel_class = st.selectbox("Travel Class", ["Economy", "Premium Economy", "Business", "First"])
        journey_day = st.selectbox("Journey Day", 
                                   ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

    with col2:
        arrival_time = st.selectbox("Arrival Time", ["Before 6 AM", "6 AM - 12 PM", "12 PM - 6 PM", "After 6 PM"])
        departure_time = st.selectbox("Departure Time", ["Before 6 AM", "6 AM - 12 PM", "12 PM - 6 PM", "After 6 PM"])
        duration_in_hours = st.number_input("Duration (in hours)", min_value=0.0, max_value=20.0, step=0.1)

    submitted = st.form_submit_button("Predict Fare")

# When submitted
if submitted:
    # Check for invalid duration
    if duration_in_hours == 0.0:
        st.warning("⚠️ Duration cannot be 0 hours. Please enter a valid flight duration.")
    else:
        try:
            # Create DataFrame for input
            input_data = pd.DataFrame([{
                "Journey_day": journey_day,
                "Class": travel_class,
                "Total_stops": total_stops,
                "Duration_in_hours": duration_in_hours,
                "Arrival": arrival_time,
                "Departure": departure_time
            }])

            # Preprocess input data
            input_processed = preprocess_custom_input(input_data, feature_columns)

            # Predict fare
            predicted_fare = model.predict(input_processed)[0]

            # Display result
            st.success(f"💰 **Estimated Flight Fare:** ₹{predicted_fare:,.2f}")

        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {e}")
