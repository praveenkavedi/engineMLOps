import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Hugging Face Model Hub
model_path = hf_hub_download(repo_id="Praveen-kavedi/engine-maintenance-rf", filename="best_random_forest.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Engine Predictive Maintenance
st.title("Engine Predictive Maintenance App")
st.write("This app is an internal tool for fleet operators and maintenance engineers to predict whether an engine requires maintenance based on its sensor readings.")
st.write("Kindly enter the engine sensor readings below to check whether the engine needs maintenance.")

# Collect user input
Engine_RPM = st.number_input("Engine RPM (rotations per minute)", min_value=0, max_value=3000, value=800)
Lub_Oil_Pressure = st.number_input("Lub Oil Pressure (bar)", min_value=0.0, max_value=10.0, value=3.3)
Fuel_Pressure = st.number_input("Fuel Pressure (bar)", min_value=0.0, max_value=25.0, value=6.7)
Coolant_Pressure = st.number_input("Coolant Pressure (bar)", min_value=0.0, max_value=10.0, value=2.3)
Lub_Oil_Temperature = st.number_input("Lub Oil Temperature (°C)", min_value=60.0, max_value=100.0, value=77.6)
Coolant_Temperature = st.number_input("Coolant Temperature (°C)", min_value=60.0, max_value=200.0, value=78.4)

# Save inputs into a dataframe
input_data = pd.DataFrame([{
    'Engine_RPM': Engine_RPM,
    'Lub_Oil_Pressure': Lub_Oil_Pressure,
    'Fuel_Pressure': Fuel_Pressure,
    'Coolant_Pressure': Coolant_Pressure,
    'Lub_Oil_Temperature': Lub_Oil_Temperature,
    'Coolant_Temperature': Coolant_Temperature,
}])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "requires maintenance 🔧" if prediction == 1 else "is operating normally ✅"
    st.write(f"Based on the sensor readings provided, the engine {result}.")
