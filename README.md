# 📰 FactSense

**FactSense** is a powerful NLP-based web application that uses a fine-tuned BERT model to detect whether a news article is real or fake. Built with Hugging Face Transformers and deployed using Streamlit, it provides an interactive interface for real-time fake news detection.

---

## 🚀 Features

- Fine-tuned BERT model (`bert-base-uncased`) for binary classification.
- Real-time news classification with confidence scores.
- Streamlit-powered intuitive UI.
- GPU-compatible training pipeline.
- Lightweight mode using dataset sampling for faster execution.
- Custom preprocessing and classification metrics (precision, recall, F1).

---

## 🧠 Model Overview

- Model: `BertForSequenceClassification`
- Dataset: Combined `Fake.csv` and `True.csv` from Kaggle.
- Preprocessing: Basic cleaning (punctuation, links), lowercasing, and tokenization.
- Training:
  - Train/Test split: 80/20
  - Optimized for CUDA (GPU)
  - Optional FP16 training enabled

---

## 📁 Project Structure

fake_app/
├── app.py # Streamlit web app
├── Fake.csv # Fake news dataset
├── True.csv # Real news dataset
├── requirements.txt # list of dependencies
├── saved_model/ # Trained BERT model (ignored in GitHub)
├── results # stored checkpoints (for model parameters, weights, etc)
├── .gitignore # Ignoring large files and venv
└── README.md # This file

---

## 📊 Dataset

The datasets used in this project are from Kaggle:

- [Fake.csv](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) — Articles labeled as fake.
- [True.csv](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) — Articles labeled as real.

You can download and place both files in the project directory.

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/mukul2956/FactSense.git
cd FactSense
2. Create a virtual environment

conda create -n fake_news python=3.10
conda activate fake_news
3. Install dependencies

pip install -r requirements.txt
If you're training the model:

pip install transformers datasets evaluate scikit-learn
4. Run the app

streamlit run app.py
🧪 Example Predictions
Input Text	Prediction	Confidence
"The Indian government announced a 6.5% GDP growth for the fiscal year, citing strong domestic demand." ✅ Real News	REAL: 0.99
"Bill Gates caught hiding microchips in vaccines, says anonymous source."	🚫 Fake News	FAKE: 0.88

🧾 License
This project is for educational and research purposes only.

🙋‍♂️ Author
Developed by Mukul

💡 Future Enhancements
Integration with live news API

Model explainability (e.g., LIME/SHAP)

Multilingual fake news detection

---
