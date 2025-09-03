import os
import streamlit as st
import requests

st.set_page_config(page_title="📰 FactSense – Fake News Detector")

# Load Hugging Face API token from environment
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/mukul-237/factsense"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Preprocessing function
import re
def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip().lower()

st.title("📰 FactSense – Fake News Detector")

user_input = st.text_area("Enter a news headline or article:", height=200)

if st.button("Detect"):
    if user_input.strip():
        text = preprocess_text(user_input)
        payload = {"inputs": text}

        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()

            # HF inference API returns a list of dicts like [{"label": "LABEL_0", "score": 0.85}]
            label = result[0]["label"]
            score = result[0]["score"]

            # Map label numbers to readable output
            if label in ["LABEL_1", "REAL"]:
                st.success(f"✅ Real News (confidence: {score:.2f})")
            else:
                st.error(f"🚫 Fake News (confidence: {score:.2f})")

        except requests.exceptions.RequestException as e:
            st.error(f"Error contacting Hugging Face API: {e}")

    else:
        st.warning("Please enter text for analysis.")
