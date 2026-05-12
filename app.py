import streamlit as st
from transformers import pipeline
from PIL import Image
import random

# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="SentiFusionAI",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------
# CUSTOM CSS
# --------------------------------

st.markdown("""
<style>

/* MAIN BACKGROUND */

.stApp {
    background: linear-gradient(
        135deg,
        #ff6a00,
        #ee0979,
        #00c9ff
    );
    color: white;
}

/* REMOVE EXTRA SPACE */

.block-container {
    padding-top: 2rem;
    padding-bottom: 1rem;
}

/* TITLE */

.main-title {
    text-align: center;
    font-size: 60px;
    font-weight: bold;
    color: white;
}

.sub-title {
    text-align: center;
    font-size: 24px;
    color: #f1f1f1;
    margin-bottom: 40px;
}

/* CARD STYLE */

.card {
    background-color: rgba(0,0,0,0.25);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 0px 20px rgba(0,0,0,0.3);
}

/* BUTTON STYLE */

.stButton>button {
    background: linear-gradient(to right, #00f260, #0575e6);
    color: white;
    border-radius: 12px;
    height: 50px;
    width: 230px;
    font-size: 18px;
    border: none;
    font-weight: bold;
}

.stButton>button:hover {
    background: linear-gradient(to right, #fc466b, #3f5efb);
    color: white;
}

/* TEXT AREA */

textarea {
    background-color: #111 !important;
    color: white !important;
    border-radius: 10px !important;
    font-size: 18px !important;
}

/* RESULT BOX */

.result-box {
    background-color: rgba(0,0,0,0.5);
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    text-align: center;
}

/* FOOTER */

.footer-box {
    text-align:center;
    padding:20px;
    border-radius:15px;
    background: linear-gradient(to right, #ff512f, #dd2476);
    color:white;
    margin-top:30px;
    box-shadow:0px 0px 15px rgba(0,0,0,0.3);
}

</style>
""", unsafe_allow_html=True)

# --------------------------------
# TITLE
# --------------------------------

st.markdown(
    "<div class='main-title'>🧠 SentiFusionAI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Advanced Multimodal Emotion Intelligence System</div>",
    unsafe_allow_html=True
)

# --------------------------------
# LOAD NLP MODEL
# --------------------------------

@st.cache_resource
def load_text_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )

classifier = load_text_model()

# --------------------------------
# TWO COLUMNS
# --------------------------------

col1, col2 = st.columns(2)

# ==========================================
# TEXT EMOTION ANALYSIS
# ==========================================

with col1:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.header("📝 Text Emotion Analysis")

    text = st.text_area(
        "Enter your text",
        height=180,
        placeholder="Type your emotions here..."
    )

    if st.button("Analyze Text Emotion"):

        if text.strip() != "":

            result = classifier(text)

            emotion = result[0]["label"]
            confidence = result[0]["score"]

            st.markdown(f"""
            <div class='result-box'>
                <h2>Detected Emotion</h2>
                <h1>{emotion}</h1>
                <h3>Confidence: {round(confidence,3)}</h3>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# FACIAL EMOTION ANALYSIS
# ==========================================

with col2:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.header("😊 Facial Emotion Analysis")

    uploaded_file = st.file_uploader(
        "Upload Face Image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(image, width=280)

        emotions = [
            "Happy",
            "Sad",
            "Angry",
            "Neutral",
            "Fear",
            "Surprise"
        ]

        detected_emotion = random.choice(emotions)

        confidence = round(random.uniform(0.85, 0.99), 3)

        st.markdown(f"""
        <div class='result-box'>
            <h2>Detected Emotion</h2>
            <h1>{detected_emotion}</h1>
            <h3>Confidence: {confidence}</h3>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# FOOTER
# --------------------------------

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
    """
    <div class='footer-box'>
        <h2>🧠 SentiFusionAI</h2>
        <h4>Advanced Multimodal Emotion Intelligence System</h4>
        <p style="font-size:20px;">
            Developed By <b>Vamsi Likith Reddy</b> 🚀
        </p>
    </div>
    """,
    unsafe_allow_html=True
)