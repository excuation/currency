import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load CNN Model
@st.cache_resource  # Cache the model to load it only once
def load_cnn_model():
    try:
        model = load_model("my_model.keras")
        st.success("✅ Model Loaded Successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_cnn_model()

# Function to preprocess image
def preprocess_image(image):
    try:
        image = image.convert("RGB")  # Ensure 3-channel image
        image = image.resize((224, 224))  # Resize to match model input
        image = np.array(image, dtype=np.float32) / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        return None

# Streamlit UI
st.title("🖼️ CNN Image Classifier")

uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # Process and predict
    processed_image = preprocess_image(image)

    if model is not None and processed_image is not None:
        try:
            st.write(f"📏 Processed Image Shape: {processed_image.shape}")  # Debugging step

            prediction = model.predict(processed_image)  # Ensure input shape matches
            predicted_class = np.argmax(prediction)  # Get predicted class

            # Display prediction
            st.success(f"🎯 Predicted Class: {predicted_class}")
        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")
    else:
        st.error("❌ Failed to process image or model not loaded. Try again.")
