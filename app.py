
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Melanoma Skin Cancer Detection",
    page_icon="🔬",
    layout="wide"
)

IMG_SIZE = (160, 160)
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best_melanoma_model.keras"
)

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# App heading
st.title("🔬 Melanoma Skin Cancer Detection")
st.write(
    "Upload a skin lesion image to see the prediction "
    "from your trained CNN model."
)

st.info(
    "Educational project only. This model is not a medical "
    "diagnostic tool. Consult a qualified dermatologist "
    "for any concerning skin lesion."
)

# Check model file
if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Please upload "
        "'best_melanoma_model.keras' to the same GitHub "
        "repository folder as app.py."
    )
    st.stop()

# Load model safely
try:
    model = load_model()
    st.success("Trained model loaded successfully!")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# Image upload
st.subheader("Upload a Skin Image")

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG or PNG image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Uploaded skin image",
                use_container_width=True
            )

        # Prepare image for prediction
        resized_image = image.resize(IMG_SIZE)
        image_array = np.array(
            resized_image,
            dtype=np.float32
        )
        image_array = np.expand_dims(image_array, axis=0)

        with st.spinner("Analyzing image..."):
            probability = float(
                model.predict(image_array, verbose=0)[0][0]
            )

        # Training folder class order:
        # benign = 0, malignant = 1
        with col2:
            st.subheader("Model Prediction")

            if probability >= 0.5:
                prediction = "Malignant"
                confidence = probability * 100
            else:
                prediction = "Benign"
                confidence = (1 - probability) * 100

            st.metric("Predicted class", prediction)
            st.metric(
                "Model score for predicted class",
                f"{confidence:.2f}%"
            )

            st.caption(
                "This score is not a validated probability "
                "of cancer and must not guide medical decisions."
            )

        st.warning(
            "The prediction may be wrong. Do not use this result "
            "to rule out or confirm skin cancer."
        )

    except Exception as e:
        st.error(f"Image processing failed: {e}")

st.divider()
st.caption(
    "Melanoma Detection | CNN image classification project"
)
