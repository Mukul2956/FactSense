import sys
import os
from dotenv import load_dotenv
import requests
import streamlit as st
import pandas as pd
import re
import torch
from sklearn.metrics import classification_report
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
st.set_page_config(page_title="📰 Fake News Detector")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")


def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip().lower()


# ...existing code for training and local model loading...

st.title("📰 FactSense ")

user_input = st.text_area("Enter a news headline or article:", height=200)

load_dotenv()
API_URL = "https://api-inference.huggingface.co/models/mukul-237/factsense"
API_TOKEN = os.getenv("HF_API_TOKEN")


def query_hf_api(text):
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


if st.button("Detect"):
    if user_input.strip():
        text = preprocess_text(user_input)
        try:
            result = query_hf_api(text)
            if isinstance(result, list) and len(result) == 2:
                real_score = next(
                    (x["score"] for x in result if "REAL" in x["label"].upper()), None
                )
                fake_score = next(
                    (x["score"] for x in result if "FAKE" in x["label"].upper()), None
                )
                label = "REAL" if (real_score or 0) > (fake_score or 0) else "FAKE"
                st.write(
                    f"Prediction confidence – REAL: {real_score:.3f}, FAKE: {fake_score:.3f}"
                )
                if max(real_score or 0, fake_score or 0) > 0.7:
                    st.success("✅ Real News" if label == "REAL" else "🚫 Fake News")
                else:
                    st.warning("🤔 Model uncertain – review with caution")
            else:
                st.error(f"API Error or unexpected response: {result}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error contacting Hugging Face API: {e}")
    else:
        st.warning("Please enter text for analysis.")
