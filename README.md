# 🟢 Persian Named Entity Recognition (NER) using Transformers + WikiAnn

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" />
  <img src="https://img.shields.io/badge/FastAPI-Framework-green?logo=fastapi" />
  <img src="https://img.shields.io/badge/Transformers-HuggingFace-yellow?logo=huggingface" />
  <img src="https://img.shields.io/badge/DeepLearning-PyTorch-EE4C2C?logo=pytorch" />
  <img src="https://img.shields.io/badge/NLP-BERT-important?logo=bert" />
  <img src="https://img.shields.io/badge/ML-ScikitLearn-F7931E?logo=scikit-learn" />
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" />
</p>


This project fine-tunes a Transformer-based model for **Named Entity Recognition (NER)** on Persian (Farsi) text using the [WikiAnn-fa dataset](https://huggingface.co/datasets/wikiann).

---

## 📌 Table of Contents

- [📖 Overview](#-overview)
- [📁 Project Structure](#-project-structure)
- [📊 Model Evaluation](#-model-evaluation)
- [⚙️ Installation](#️-installation)
  - [Clone the repository](#clone-the-repository)
  - [🧪 Create a virtual environment and install dependencies](#-create-a-virtual-environment-and-install-dependencies)
- [🚀 Usage](#-usage)
  - [▶ Run prediction script](#-run-prediction-script)
  - [🧪 Run FastAPI App](#-run-fastapi-app)
  - [🧪 Test the API (POST request)](#-test-the-api-post-request)
  - [✅ Model Predictions](#-model-predictions)
- [🧠 Model](#-model)
- [📬 Contact](#-contact)

---

## 📖 Overview

**Goal:** Train a NER model for the Persian language to extract entities such as:

- `PER` → Person
- `LOC` → Location
- `ORG` → Organization

This is done by fine-tuning a pretrained model using 🤗 Hugging Face Transformers on the WikiAnn-fa dataset.

---

## 📁 Project Structure

```bash
NER-WIKIANN-FA/
├── data/
│   ├── results.csv              # Output predictions
│   └── sample_data.txt          # Optional input sample(s)

├── ner_model/                   # Fine-tuned model and tokenizer
│   ├── config.json
│   ├── labels.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.txt
├── app.py     # FastAPI
├── notebooks/
│   └── model_ner_wikiann-fa.ipynb   # Notebook for training and evaluation

├── predict.py                   # Script for running predictions
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md                    # Project documentation
```


## 📊 Model Evaluation

| **Metric**        | **Validation Set** | **Test Set** |
| ----------------- | ------------------ | ------------ |
| **Loss**          | 0.1648             | 0.1794       |
| **Precision**     | 0.9351             | 0.9384       |
| **Recall**        | 0.9429             | 0.9431       |
| **F1 Score**      | 0.9390             | 0.9407       |
| **Accuracy**      | 0.9729             | 0.9720       |
| **Runtime (sec)** | 12.02              | 11.34        |
| **Steps/sec**     | 51.99              | 55.11        |
| **Epochs**        | 5                  | 5            |

📁 **Source:** `data/results.csv`

## ⚙️ Installation

### Clone the repository:

```bash
git clone https://github.com/mehdighelich1379/persian-ner-nlp.git
cd persian-ner-nlp
```

### 🧪 Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Usage

### ▶ Run prediction script

Predict entities from a single sentence:

```bash
python predict.py
```

Or modify `sample_data.txt` and run:

```bash
python predict.py --input_file data/sample_data.txt
```


### 🧪 Run FastAPI App

Start the API server:

```bash
uvicorn app:app --reload
```

### 🧪 Test the API (POST request)

Send a JSON request like below to the `/predict/` endpoint using Postman or curl:

```json
{
  "text": "محمدرضا شریفی‌نیا در سال ۱۳۹۸ به همراه تیم تحقیقات انرژی دانشگاه صنعتی شریف، سفری به آلمان داشت و در کنفرانسی که در شهر برلین توسط شرکت زیمنس برگزار شد، درباره نقش ایران در بازار جهانی گاز سخنرانی کرد."
}
```

### ✅ Model Predictions
```json
{
  "entities": [
    { "token": "محمدرضا", "label": "B-PER" },
    { "token": "شریفینیا", "label": "I-PER" },
    { "token": "در", "label": "O" },
    { "token": "سال", "label": "O" },
    { "token": "۱۳۹۸", "label": "O" },
    { "token": "به", "label": "O" },
    { "token": "همراه", "label": "O" },
    { "token": "تیم", "label": "O" },
    { "token": "تحقیقات", "label": "I-ORG" },
    { "token": "انرژی", "label": "O" },
    { "token": "دانشگاه", "label": "B-ORG" },
    { "token": "صنعتی", "label": "I-ORG" },
    { "token": "شریف", "label": "I-ORG" },
    { "token": "،", "label": "O" },
    { "token": "سفری", "label": "O" },
    { "token": "به", "label": "O" },
    { "token": "المان", "label": "B-LOC" },
    { "token": "داشت", "label": "O" },
    { "token": "و", "label": "O" },
    { "token": "در", "label": "O" },
    { "token": "کنفرانسی", "label": "O" },
    { "token": "که", "label": "O" },
    { "token": "در", "label": "O" },
    { "token": "شهر", "label": "O" },
    { "token": "برلین", "label": "B-LOC" },
    { "token": "توسط", "label": "O" },
    { "token": "شرکت", "label": "O" },
    { "token": "زیمنس", "label": "O" },
    { "token": "برگزار", "label": "O" },
    { "token": "شد", "label": "O" },
    { "token": "،", "label": "O" },
    { "token": "درباره", "label": "O" },
    { "token": "نقش", "label": "O" },
    { "token": "ایران", "label": "B-LOC" },
    { "token": "در", "label": "O" },
    { "token": "بازار", "label": "O" },
    { "token": "جهانی", "label": "I-ORG" },
    { "token": "گاز", "label": "O" },
    { "token": "سخنرانی", "label": "O" },
    { "token": "کرد", "label": "O" },
    { "token": ".", "label": "O" }
  ]
}
```

## 🧠 Model

- ✅ **Base model**: `bert-base-parsbert-uncased`
- ✅ **Task**: Named Entity Recognition (NER)
- ✅ **Language**: Persian (Farsi)
- ✅ **Training data**: [WikiAnn-fa dataset](https://huggingface.co/datasets/wikiann)
- ✅ **Fine-tuned** on Persian NER using the BIO tagging scheme


## 📬 Contact

If you have any questions, suggestions, or feedback, feel free to reach out:

📧 Email: [qelejkhanimehdi@gmail.com](mailto:qelejkhanimehdi@gmail.com)

