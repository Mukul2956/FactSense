import os
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="📰 FactSense – Fake News Detector")

st.title("📰 FactSense – Fake News Detector")

# Hugging Face API token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "mukul-237/factsense"

if HF_TOKEN is None:
    st.error("HF_TOKEN is not set. Please configure the environment variable.")
    st.stop()

# Initialize HF pipeline (will automatically use CPU on Render)
@st.cache_resource(show_spinner=False)
def get_classifier():
    return pipeline(
        "text-classification",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        device=-1,  # CPU
        use_auth_token=HF_TOKEN
    )

classifier = get_classifier()

# Text input
user_input = st.text_area("Enter a news headline or article:", height=200)

if st.button("Detect"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            result = classifier(user_input[:512])  # limit to first 512 chars
            label = result[0]["label"]
            score = result[0]["score"]

        st.write(f"Prediction confidence – {label}: {score:.3f}")
        if score > 0.7:
            st.success("✅ Real News" if label == "LABEL_1" else "🚫 Fake News")
        else:
            st.warning("🤔 Model uncertain – review with caution")
    else:
        st.warning("Please enter text for analysis.")
