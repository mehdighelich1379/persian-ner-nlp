from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np
import json

# --- Load model and tokenizer ---
model_path = "./ner_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# --- Load label list ---
with open(f"{model_path}/labels.json", "r", encoding="utf-8") as f:
    label_list = json.load(f)

app = FastAPI(title="Persian NER API")

class InputText(BaseModel):
    text: str

@app.post("/predict/")
def predict_ner(input: InputText):
    tokens = tokenizer(input.text, return_tensors="pt", truncation=True, is_split_into_words=False)
    with torch.no_grad():
        output = model(**tokens)

    predictions = torch.argmax(output.logits, dim=2)[0].tolist()
    input_tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])

    # Match labels with tokens (ignoring special tokens like [CLS], [SEP])
    results = []
    for token, pred in zip(input_tokens, predictions):
        label = label_list[pred]
        if not token.startswith("##") and token not in tokenizer.all_special_tokens:
            results.append({"token": token, "label": label})

    return {"entities": results}
