import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Crime Analysis", layout="centered")

st.title("🚔 Crime Pattern Analysis in India")

# Load model
model = pickle.load(open("models/trained_model.pkl", "rb"))

st.subheader("Enter Crime Details")

# Inputs (only useful ones)
city = st.selectbox("City", ["Mumbai", "Delhi", "Bangalore", "Pune"])
victim_age = st.number_input("Victim Age", min_value=0, max_value=100)
gender = st.selectbox("Victim Gender", ["Male", "Female"])
weapon = st.selectbox("Weapon Used", ["Knife", "Gun", "None", "Unknown"])
police = st.number_input("Police Deployed", min_value=0)

# Create dataframe
input_data = pd.DataFrame({
    "City": [city],
    "Victim Age": [victim_age],
    "Victim Gender": [gender],
    "Weapon Used": [weapon],
    "Police Deployed": [police]
})

# Encode like training
input_data = pd.get_dummies(input_data)

# Align with training columns
model_columns = pickle.load(open("models/trained_model.pkl", "rb")).feature_names_in_

for col in model_columns:
    if col not in input_data:
        input_data[col] = 0

input_data = input_data[model_columns]

# Predict
if st.button("Predict Crime Type"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Crime Domain: {prediction[0]}")
