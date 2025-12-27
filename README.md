# Developing NER Models for Financial Data Extraction 📊

This project implements an end-to-end Natural Language Processing (NLP) pipeline to extract structured financial information from unstructured financial text such as news articles and reports.

The project was developed as part of an AI internship and demonstrates data collection, preprocessing, entity extraction, document segmentation, and structured output generation.

---

## 🎯 Project Objectives

- Collect and preprocess financial text data
- Perform tokenization, lemmatization, and POS tagging
- Extract financial entities (companies, dates, monetary values)
- Support user-defined financial metric extraction (e.g., profit, revenue)
- Handle long financial documents with section segmentation
- Detect and parse table-like numerical data
- Generate structured JSON output

---

## 🗂️ Project Structure

Infosys-Project/
│
├── data/
│ ├── raw/ # Raw Kaggle datasets
│ └── processed/ # Processed data
│
├── output/
│ └── final_output.json # Final structured output
│
├── main.py # Main Python script
├── README.md
└── .gitignore


---

## 📊 Dataset Description

The project uses financial news datasets collected from Kaggle:
- Financial News Dataset
- Indian Financial News Articles Dataset

These datasets provide a large and diverse corpus of financial text for NLP tasks.

---

## ⚙️ Technologies Used

- Python 3.10
- spaCy
- pandas
- matplotlib
- VS Code

---


## 🔄 Project Workflow

### 1. Data Collection
- Financial datasets downloaded from Kaggle
- Multiple datasets combined into a single corpus

### 2. Data Preprocessing
- Tokenization
- Lemmatization
- Part-of-Speech (POS) tagging
- Stopword removal
- Cleaning noisy financial text

### 3. Exploratory Data Analysis (EDA)
- Text length distribution analysis
- Summary statistics of financial documents

### 4. Named Entity Recognition (NER)
- Extracts organizations, dates, and monetary values
- Uses pretrained spaCy NER model

### 5. Model Training (Baseline)

A baseline NER model was fine-tuned using spaCy with auto-annotated financial text to validate entity extraction performance.  
Training was performed locally on CPU, and the trained model was tested on unseen financial sentences.

Advanced model fine-tuning using CRF or transformer-based models (e.g., FinBERT) is planned as future work.


### 6. Custom User-Defined Extraction
- Allows extraction of financial metrics such as profit or revenue based on user input
- Rule-based extraction approach

### 7. Long Document Handling
- Simulates long financial documents
- Segments documents into sections such as:
  - MD&A
  - Risk Factors
  - Financial Statements
- Detects and parses table-like numeric content

### 8. Final Output
- Generates structured JSON output containing:
  - Section-wise document content
  - Extracted named entities
  - Custom extracted financial metrics
  - Parsed table data

---

## ▶️ How to Run the Project

1. Clone the repository:
```bash
git clone https://github.com/Shailaja-poojari/Infosys-Project.git
cd Infosys-Project

2. Install required libraries:

pip install pandas matplotlib spacy
python -m spacy download en_core_web_sm


3. Run the project:

python main.py


4. View the output:

output/final_output.json

Future Enhancements

Fine-tuning financial-specific transformer models (e.g., FinBERT)

Adding evaluation metrics (precision, recall, F1-score)

Building a lightweight user interface using Streamlit

Deploying the application

Author

Developed by Shailaja Poojari as part of an AI internship project.


---

## ✅ FINAL GIT COMMANDS (JUST 3)

After updating README:

```bash
git add README.md
git commit -m "Update project title as per official documentation"
git push
