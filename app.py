import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load CNN Model
@st.cache_resource
def load_cnn_model():
    return load_model("my_model.keras")

model = load_cnn_model()

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize to match model input size
    image = np.array(image) / 255.0   # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("CNN Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process and predict
    processed_image = preprocess_image(image)
    st.write(f"Processed Image Shape: {processed_image.shape}")  # Debugging line

    try:
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        st.success(f"Predicted Class: {predicted_class}")
    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
