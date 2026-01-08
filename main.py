import pandas as pd
import spacy
import re

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load datasets
df1 = pd.read_csv("data/raw/financial_news.csv")
df2 = pd.read_csv("data/raw/indian_financial_news.csv")

# Standardize text column
df1["content"] = df1["intro"]
df2["content"] = df2["Description"]

# Keep only text
df = pd.concat([df1[["content"]], df2[["content"]]], ignore_index=True)

# Remove empty rows
df.dropna(inplace=True)

print("Combined dataset shape:", df.shape)

def preprocess_text(text):
    # Remove currency symbols
    text = re.sub(r'[$€₹]', '', text)

    doc = nlp(text)
    tokens = []

    for token in doc:
        if not token.is_stop and not token.is_punct:
            tokens.append({
                "token": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_
            })
    return tokens

# Apply preprocessing on first 5 rows only (fast & safe)
df["processed"] = df["content"].head(5).apply(preprocess_text)

print("\nSample Preprocessed Output:\n")
print(df["processed"].iloc[0])

# ---------------- EDA ----------------
df["text_length"] = df["content"].apply(lambda x: len(str(x).split()))

print("\nEDA Summary:")
print(df["text_length"].describe())

# ---------------- BASIC NER ----------------

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Apply NER on a small sample (fast & safe)
df["entities"] = df["content"].head(5).apply(extract_entities)

print("\nSample NER Output:\n")
print(df["entities"].iloc[0])

# ---------------- CUSTOM / USER-DEFINED EXTRACTION ----------------

def extract_by_keyword(text, keyword):
    results = []
    for sentence in text.split("."):
        if keyword.lower() in sentence.lower():
            results.append(sentence.strip())
    return results

# Simulating user-defined inputs
keywords = ["revenue", "profit", "assets", "liabilities"]

print("\nCustom Extraction Results:\n")

sample_text = df["content"].iloc[0]

for key in keywords:
    extracted = extract_by_keyword(sample_text, key)
    if extracted:
        print(f"{key.upper()} →", extracted)

# ---------------- MILESTONE 4: LONG DOCUMENT HANDLING ----------------

# Simulate a long document by combining multiple articles
long_document = " ".join(df["content"].head(200).tolist())

print("\nLong document length:", len(long_document))

SECTION_HEADERS = {
    "MD&A": ["management", "discussion"],
    "Risk Factors": ["risk", "uncertainty"],
    "Financial Statements": ["revenue", "profit", "assets", "liabilities"]
}

def segment_document(text):
    sections = {}
    current_section = "General"
    sections[current_section] = []

    for sentence in text.split("."):
        for section, keywords in SECTION_HEADERS.items():
            if any(k in sentence.lower() for k in keywords):
                current_section = section
                sections.setdefault(current_section, [])
        sections[current_section].append(sentence.strip())

    return sections


sections = segment_document(long_document)

print("\nDetected Sections:")
for sec in sections:
    print("-", sec)

def is_table_line(line):
    return sum(char.isdigit() for char in line) > 6
table_lines = [line for line in long_document.split(".") if is_table_line(line)]

print("\nSample Table-like Lines:")
print(table_lines[:2])

def parse_table(lines):
    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) > 1:
            rows.append({
                "item": " ".join(parts[:-1]),
                "value": parts[-1]
            })
    return rows


parsed_tables = parse_table(table_lines[:5])

print("\nParsed Table Output:")
print(parsed_tables)


import json

final_output = {
    "company": "Sample Financial Corpus",
    "sections": sections,
    "sample_entities": df["entities"].head(3).tolist(),
    "custom_extraction_example": extract_by_keyword(df["content"].iloc[0], "profit"),
    "tables": parsed_tables
}

with open("output/final_output.json", "w") as f:
    json.dump(final_output, f, indent=2)

print("\nFinal structured JSON saved to output/final_output.json")

# ===============================
# EXTRA: PROCESS ANNUAL REPORT (10-K)
# ===============================

annual_report_path = "raw/annual_reports/adp_10k_2021.txt"

try:
    with open(annual_report_path, "r", encoding="utf-8", errors="ignore") as f:
        annual_text = f.read()

    print("\nAnnual Report Loaded Successfully")
    print("Annual Report Length:", len(annual_text))

    # ---------- Section Detection ----------
    sections = {
        "General": [],
        "Financial Statements": [],
        "MD&A": [],
        "Risk Factors": []
    }

    for line in annual_text.split("\n"):
        line_lower = line.lower()

        if "risk factor" in line_lower:
            sections["Risk Factors"].append(line.strip())
        elif "management’s discussion" in line_lower or "management's discussion" in line_lower:
            sections["MD&A"].append(line.strip())
        elif "financial statement" in line_lower:
            sections["Financial Statements"].append(line.strip())

    print("\nDetected Sections from Annual Report:")
    for k, v in sections.items():
        print(f"{k}: {len(v)} lines")

except FileNotFoundError:
    print("Annual report file not found. Skipping annual report processing.")

    # =========================================================
# FINAL ENHANCEMENTS: COMPANY NAME & RISK FACTOR FALLBACK
# =========================================================

# ---------- COMPANY NAME EXTRACTION ----------
def extract_company_names(text, top_n=3):
    doc = nlp(text)
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    return list(dict.fromkeys(orgs))[:top_n]  # unique + ordered


company_names = extract_company_names(long_document)

if not company_names:
    company_names = ["Unknown Company"]


# ---------- RISK FACTOR FALLBACK (NEWS DATA) ----------
RISK_KEYWORDS = [
    "risk", "uncertain", "challenge", "threat",
    "slowdown", "regulatory", "litigation",
    "debt", "loss", "decline", "volatility"
]

risk_fallback = []

for sentence in long_document.split("."):
    if any(k in sentence.lower() for k in RISK_KEYWORDS):
        cleaned = sentence.strip()
        if len(cleaned) > 40:
            risk_fallback.append(cleaned)

risk_fallback = risk_fallback[:5]

# If Risk Factors section is empty, fill it
if "Risk Factors" in final_output["sections"]:
    if len(final_output["sections"]["Risk Factors"]) == 0:
        final_output["sections"]["Risk Factors"] = risk_fallback


# ---------- UPDATE FINAL JSON ----------
final_output["company"] = company_names
final_output["data_sources"] = [
    "Financial News Dataset (Kaggle)",
    "Indian Financial News Dataset (Kaggle)",
    "SEC 10-K Annual Report (Text)"
]

with open("output/final_output.json", "w") as f:
    json.dump(final_output, f, indent=2)

print("\n✔ Enhanced JSON updated with company name and risk factors")
