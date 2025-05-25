import sys
print(sys.executable)

import streamlit as st
import pandas as pd
import re
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

from sklearn.metrics import classification_report
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import evaluate

st.set_page_config(page_title="📰 Fake News Detector")

metric = evaluate.load("accuracy")

# Device check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.write(f"Using device: {device}")

# Preprocessing
def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip().lower()

# Load and preprocess
def load_data():
    df_fake = pd.read_csv("Fake.csv")
    df_true = pd.read_csv("True.csv")
    df_fake['label'] = 0
    df_true['label'] = 1
    df = pd.concat([df_fake, df_true], ignore_index=True).sample(frac=0.2, random_state=42)
    df['text'] = (df['title'] + " " + df['text']).apply(preprocess_text)
    return Dataset.from_pandas(df[['text', 'label']].dropna())

# Tokenize
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

# Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    report = classification_report(labels, preds, output_dict=True)
    return {
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
    }

# Train function
def train_model():
    dataset = load_data()
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    split = dataset.train_test_split(test_size=0.2)
    train_ds, eval_ds = split["train"], split["test"]

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.to(device)

    args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model.save_pretrained("saved_model")
    tokenizer.save_pretrained("saved_model")
    return model, tokenizer

# App UI

st.title("📰 Fake News Detection (Improved)")

if st.button("Train Model"):
    with st.spinner("Training..."):
        model, tokenizer = train_model()
    st.success("Training complete!")
else:
    try:
        tokenizer = BertTokenizerFast.from_pretrained("saved_model")
        model = BertForSequenceClassification.from_pretrained("saved_model")
        model.to(device)
        model.eval()
    except Exception as e:
        st.error("No saved model found. Please train the model first.")
        st.stop()

user_input = st.text_area("Enter a news headline or article:", height=200)
if st.button("Detect"):
    if user_input.strip():
        text = preprocess_text(user_input)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1).flatten()
            label = torch.argmax(probs).item()

        st.write(f"Prediction confidence – REAL: {probs[1]:.3f}, FAKE: {probs[0]:.3f}")
        if probs[label] > 0.7:
            st.success("✅ Real News" if label == 1 else "🚫 Fake News")
        else:
            st.warning("🤔 Model uncertain – review with caution")
    else:
        st.warning("Please enter text for analysis.")
