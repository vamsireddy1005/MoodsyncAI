import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from transformers import pipeline
import torch

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="SentiFusionAI",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #ff4b4b,
        #ff416c,
        #7f00ff,
        #00c6ff
    );
    background-attachment: fixed;
    color: white;
}

.big-title {
    font-size: 60px;
    font-weight: bold;
    text-align: center;
    color: white;
    margin-top: 20px;
}

.sub-title {
    font-size: 28px;
    text-align: center;
    color: white;
    margin-bottom: 40px;
}

.box {
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(10px);
    padding: 25px;
    border-radius: 20px;
    margin-top: 20px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}

.footer-box {
    text-align: center;
    padding: 30px;
    color: white;
    margin-top: 40px;
}

.stButton>button {
    background: linear-gradient(to right, #ff416c, #7f00ff);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: bold;
}

.stTextArea textarea {
    background-color: rgba(255,255,255,0.1);
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ---------------- #

@st.cache_resource
def load_cnn_model():
    model = load_model("models/sentifusion_cnn_model.h5")
    return model

@st.cache_resource
def load_text_model():
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    return classifier

model = load_cnn_model()
classifier = load_text_model()

# ---------------- HEADER ---------------- #

st.markdown(
    "<div class='big-title'>🧠 SentiFusionAI</div>",
    unsafe_allow_html=True
)

st.markdown("""
<div class='sub-title'>
Advanced Multimodal Emotion Intelligence System
</div>
""", unsafe_allow_html=True)

# ---------------- LAYOUT ---------------- #

col1, col2 = st.columns(2)

# ==========================================================
# TEXT EMOTION ANALYSIS
# ==========================================================

with col1:

    st.markdown("<div class='box'>", unsafe_allow_html=True)

    st.header("📝 Text Emotion Analysis")

    user_text = st.text_area("Enter Text")

    if st.button("Analyze Text Emotion"):

        if user_text.strip() != "":

            result = classifier(user_text)[0]

            label = result['label']
            score = result['score'] * 100

            st.success(f"Emotion: {label}")
            st.info(f"Confidence Score: {score:.2f}%")

        else:
            st.warning("Please enter text.")

    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# FACIAL EMOTION ANALYSIS
# ==========================================================

with col2:

    st.markdown("<div class='box'>", unsafe_allow_html=True)

    st.header("😊 Facial Emotion Analysis")

    uploaded_file = st.file_uploader(
        "Upload Face Image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:

        # Open image
        image = Image.open(uploaded_file).convert("RGB")

        # Display uploaded image
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        # Convert image to numpy
        img = np.array(image)

        # Resize image
        face = cv2.resize(img, (48, 48))

        # Normalize
        face = face.astype("float32") / 255.0

        # Expand dimensions
        face = np.expand_dims(face, axis=0)

        # Predict emotion
        prediction = model.predict(face)

        emotion_labels = [
            "Angry",
            "Disgust",
            "Fear",
            "Happy",
            "Neutral",
            "Sad",
            "Surprise"
        ]

        predicted_emotion = emotion_labels[np.argmax(prediction)]

        confidence = np.max(prediction) * 100

        st.success(f"Predicted Emotion: {predicted_emotion}")

        st.info(f"Confidence Score: {confidence:.2f}%")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ---------------- #

st.markdown("""
<div class='footer-box'>

<h2>🧠 SentiFusionAI</h2>

<h4>Advanced Multimodal Emotion Intelligence System</h4>

<p style="font-size:20px;">
Developed By <b>Vamsi Likith Reddy Mallidi</b> 🚀
</p>

</div>
""", unsafe_allow_html=True) 
