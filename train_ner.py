import random
import spacy
from spacy.training import Example
import pandas as pd
import re
from pathlib import Path

# ---------------------------
# 1. LOAD BASE MODEL
# ---------------------------
nlp = spacy.load("en_core_web_sm")

# ---------------------------
# 2. LOAD DATA
# ---------------------------
df = pd.read_csv("data/raw/financial_news.csv")  # change name if needed
texts = df.iloc[:, 0].dropna().astype(str).tolist()

# Take small subset (fast training)
texts = texts[:300]

# ---------------------------
# 3. AUTO LABEL FUNCTION
# ---------------------------
def auto_annotate(text):
    entities = []

    # MONEY patterns
    for m in re.finditer(r'Rs\.?\s?\d+[,\d]*', text):
        entities.append((m.start(), m.end(), "MONEY"))

    # ORG patterns (simple heuristic)
    for org in ["TCS", "Infosys", "ICICI", "HDFC", "Reliance", "REC", "BSE"]:
        for m in re.finditer(org, text):
            entities.append((m.start(), m.end(), "ORG"))

    return (text, {"entities": entities})

train_data = [auto_annotate(t) for t in texts if len(auto_annotate(t)[1]["entities"]) > 0]

print("Training samples:", len(train_data))

# ---------------------------
# 4. TRAIN NER
# ---------------------------
ner = nlp.get_pipe("ner")

optimizer = nlp.resume_training()

for epoch in range(5):
    random.shuffle(train_data)
    losses = {}
    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], sgd=optimizer, losses=losses)
    print(f"Epoch {epoch+1}, Losses: {losses}")

# ---------------------------
# 5. SAVE MODEL
# ---------------------------
output_dir = Path("trained_model")
nlp.to_disk(output_dir)
print("Model saved to 'trained_model'")
