import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_dir = "./ner_model"

# tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForTokenClassification.from_pretrained(model_dir)


with open(f"{model_dir}/labels.json", "r", encoding="utf-8") as f:
    label_list = json.load(f)


def predict_ner(sentence, tokenizer, model, label_list):


    tokens = sentence.split()
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True)
    word_ids = inputs.word_ids()

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)[0].numpy()


    final_tokens, final_tags = [], []
    prev_word_id = None
    for i, word_id in enumerate(word_ids):
        if word_id is None or word_id == prev_word_id:
            continue
        final_tokens.append(tokens[word_id])
        final_tags.append(label_list[predictions[i]])
        prev_word_id = word_id

    for token, tag in zip(final_tokens, final_tags):
        print(f"{token:10} → {tag}")




def predict_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line:
            print(f"\n📝 Sentence: {line}")
            predict_ner(line, tokenizer, model, label_list)


predict_from_file("data/sample_data.txt")
