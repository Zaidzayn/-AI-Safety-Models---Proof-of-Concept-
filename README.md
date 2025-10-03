# 🏆 AI Safety Models - Proof of Concept

This repository contains a Proof of Concept (POC) for a suite of AI Safety Models designed for a conversational AI platform. The project demonstrates a scalable, multi-layered system for identifying and mitigating harmful content in real-time, featuring a user-friendly interface built with Streamlit.

---

## ✨ Features

This POC implements a safety system with the following core models, which can be selected at runtime:

* **Abuse Language Detection:** A fine-tuned Transformer model (`RoBERTa` for English, `XLM-RoBERTa` for multilingual) to identify toxic or inappropriate content with high accuracy.
* **Escalation Pattern Recognition:** An advanced sequence model (Sentence-Transformer + LSTM) that analyzes conversation history to detect patterns of intensifying negativity.
* **Content Filtering:** A rule-based system that uses the abuse model's score in conjunction with a user's age profile to filter age-inappropriate content.

The application features a sidebar to seamlessly switch between the English-Only and Multilingual model suites for direct comparison.

---

## ⚙️ System Architecture

The system is built around a central **`SafetyOrchestrator`** class, which acts as a modular hub for the different safety models. This design allows for easy extension and upgrading of individual components.

**Data Flow:**
`User Input → Streamlit UI → SafetyOrchestrator → [Abuse Model, Escalation Model, Content Filter] → Final Decision`

---

## 📂 Project Structure

```text
ai_safety_poc/
├── app.py                      # The main Streamlit web application
├── data/
│   ├── raw/                    # Raw datasets (user must download)
│   └── processed/              # Cleaned, split datasets
├── models/
│   ├── abuse-detector-roberta-lora/
│   ├── abuse-detector-xlm-roberta-lora/
│   └── escalation-detector-lstm/
├── src/
│   ├── data_preprocessing.py
│   ├── escalation_model_def.py
│   ├── evaluate_model.py
│   ├── model_training_multi.py
│   ├── orchestrators.py
│   ├── prepare_escalation_data.py
│   └── train_escalation_model.py
├── tests/
│   └── test_edge_cases.py
├── .gitignore
├── requirements.txt
└── README.md
##  Setup & Installation

## 🚀 Setup & Installation

**1. Clone the Repository**
```bash
git clone https://github.com/Zaidzayn/-AI-Safety-Models---Proof-of-Concept-
cd ai_safety_poc


2. Create and Activate a Virtual Environment

Bash

# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies

Bash

pip install -r requirements.txt
4. Download Datasets

Place the required dataset in the data/raw/ directory.

Jigsaw Dataset: Download train.csv from the Jigsaw Toxic Comment Classification Challenge and place it at data/raw/train.csv.

5. Pre-trained Models

This repository requires trained model files in the models/ directory. You can train them yourself by running the scripts in the src/ folder on a GPU environment like Google Colab.

USAGE
1. Prepare the Data

Run the preprocessing script to clean and split the datasets.

Bash

python src/data_preprocessing.py
2. Run the Streamlit Demo

This command launches the interactive web application.

Bash

streamlit run app.py
3. Run Tests

To run the automated tests, use pytest.

Bash

pytest
