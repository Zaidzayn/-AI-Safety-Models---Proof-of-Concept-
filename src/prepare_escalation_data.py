import pandas as pd
import random

# --- Configuration ---
JIGSAW_PATH = "data/processed/abuse_train.csv" # Useing our processed, clean data
OUTPUT_PATH = "data/processed/escalation_dataset.csv"
NUM_SAMPLES = 5000  # Create 5000 escalating and 5000 normal examples
SEQUENCE_LENGTH = 5

def create_escalation_dataset():
    """Synthesizeing a dataset for training the escalation detection model."""
    print("Synthesizing escalation dataset...")
    
    try:
        df = pd.read_csv(JIGSAW_PATH)
    except FileNotFoundError:
        print(f"Error: Make sure {JIGSAW_PATH} exists. Run data_preprocessing.py first.")
        return

    benign_texts = df[df['label'] == 0]['text'].tolist()
    toxic_texts = df[df['label'] == 1]['text'].tolist()

    if len(benign_texts) < 100 or len(toxic_texts) < 100:
        print("Error: Not enough benign or toxic texts to generate a dataset.")
        return

    dataset = []

    # Createing "Escalating" samples
    for _ in range(NUM_SAMPLES):
        # Example: 2 benign messages followed by 3 toxic messages
        sequence = (
            random.sample(benign_texts, 2) + 
            random.sample(toxic_texts, 3)
        )
        random.shuffle(sequence) # Shuffle to add variety
        # Combine into a single text entry, separated by a special token
        dataset.append({"text_sequence": " [SEP] ".join(sequence), "label": 1})

    # Createing "Normal" samples
    for _ in range(NUM_SAMPLES):
        sequence = random.sample(benign_texts, SEQUENCE_LENGTH)
        dataset.append({"text_sequence": " [SEP] ".join(sequence), "label": 0})

    # Create and save the final DataFrame
    final_df = pd.DataFrame(dataset)
    final_df = final_df.sample(frac=1).reset_index(drop=True) # Shuffle the whole dataset
    final_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"✅ Successfully created escalation dataset with {len(final_df)} samples.")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    create_escalation_dataset()