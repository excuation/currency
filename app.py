import streamlit as st
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("my_model.keras")

# Function to make predictions
def predict(input_data):
    input_data = np.array(input_data).reshape(1, -1)  # Reshape input
    prediction = model.predict(input_data)
    return prediction

# Streamlit UI
st.title("Deep Learning Model Deployment")

# User input
input_values = st.text_input("Enter comma-separated input values:")

if st.button("Predict"):
    try:
        input_list = [float(i) for i in input_values.split(",")]
        result = predict(input_list)
        st.success(f"Prediction: {result}")
    except:
        st.error("Invalid input! Please enter numerical values.")
