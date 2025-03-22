import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st

# Load the trained model
model = tf.keras.models.load_model("my_model.keras")


# Define class labels (Modify based on your dataset)
class_labels = {0: "₹50", 1: "₹100", 2: "₹200", 3: "₹500", 4: "₹2000"}

# Preprocess uploaded image
def preprocess_image(image):
    image = Image.open(image)
    image = image.resize((128, 128))  # Ensure correct size
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("💰 Indian Currency Note Classifier")
st.write("Upload an image of an Indian currency note, and the model will predict its denomination.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess and predict
    image = preprocess_image(uploaded_file)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels.get(predicted_class, "Unknown")

    # Display results
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write(f"✅ **Predicted Currency Note:** {predicted_label}")
