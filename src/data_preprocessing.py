import pandas as pd
from sklearn.model_selection import train_test_split
import re
import os

# --- Configuration ---
JIGSAW_DATA_PATH = "data/raw/train.csv"
CRISIS_DATA_PATH = "data/raw/suicide_detection.csv" 
OUTPUT_PATH = "data/processed/"

# --- Reusable Functions ---
def clean_text(text):
    """Applies basic text cleaning."""
    # Ensure text is a string
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Simplified cleaner
    return text

# --- Dataset-Specific Processors ---
def process_jigsaw_data():
    """Cleans and splits the Jigsaw dataset for abuse detection."""
    print("➡️ Processing Jigsaw dataset for ABUSE detection...")
    if not os.path.exists(JIGSAW_DATA_PATH):
        print(f"Warning: Jigsaw data not found at {JIGSAW_DATA_PATH}. Skipping.")
        return

    df = pd.read_csv(JIGSAW_DATA_PATH)
    
    toxic_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df['label'] = df[toxic_cols].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    df = df[['comment_text', 'label']].rename(columns={'comment_text': 'text'})
    
    df['text'] = df['text'].apply(clean_text)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['label'])
    
    train_df.to_csv(f"{OUTPUT_PATH}abuse_train.csv", index=False)
    val_df.to_csv(f"{OUTPUT_PATH}abuse_val.csv", index=False)
    test_df.to_csv(f"{OUTPUT_PATH}abuse_test.csv", index=False)
    print("   ✅ Abuse data saved to data/processed/")

def process_crisis_data():
    """Cleans and splits the dataset for crisis detection."""
    print("\n➡️ Processing dataset for CRISIS detection...")
    if not os.path.exists(CRISIS_DATA_PATH):
        print(f"Warning: Crisis data not found at {CRISIS_DATA_PATH}. Skipping.")
        return

    df = pd.read_csv(CRISIS_DATA_PATH)
    
    # The dataset has 'text' and 'class' columns. We need to convert 'class' to a numeric label.
    # Convert 'suicide' to 1, and 'non-suicide' to 0.
    df['label'] = df['class'].apply(lambda x: 1 if x == 'suicide' else 0)
    df = df[['text', 'label']] # Keep only the columns we need
    
    df['text'] = df['text'].apply(clean_text)
    df = df.dropna(subset=['text']) 
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['label'])
    
    # Save with a new prefix to keep them separate
    train_df.to_csv(f"{OUTPUT_PATH}crisis_train.csv", index=False)
    val_df.to_csv(f"{OUTPUT_PATH}crisis_val.csv", index=False)
    test_df.to_csv(f"{OUTPUT_PATH}crisis_test.csv", index=False)
    print("   ✅ Crisis data saved to data/processed/")

# --- Main Execution Block ---
if __name__ == '__main__':
    # Ensure the output directory exists
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    
    process_jigsaw_data()
    process_crisis_data()
    
    print("\nAll datasets processed successfully!")