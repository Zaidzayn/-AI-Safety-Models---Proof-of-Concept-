import torch
import pandas as pd
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def evaluate_model():
    """Loads the fine-tuned model and evaluates its performance on the test set."""
    # --- 1. Configuration ---
    base_model_name = "roberta-base"
    
    # Build a robust path to the model and data
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(project_root, "models", "abuse-detector-roberta-lora")
    test_data_path = os.path.join(project_root, "data", "processed", "abuse_test.csv")

    print("--- Model Evaluation ---")
    print(f"Loading model from: {model_path}")
    print(f"Loading test data from: {test_data_path}")

    # --- 2. Load Model and Tokenizer ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
    model = PeftModel.from_pretrained(base_model, model_path).to(device)
    model.eval()

    # --- 3. Load Test Data ---
    test_df = pd.read_csv(test_data_path).dropna(subset=['text'])
    
    # Using a smaller subset for quick evaluation if needed
    # test_df = test_df.sample(n=1000, random_state=42)

    true_labels = []
    predictions = []

    # --- 4. Performing Inference on Test Set ---
    print("\nRunning predictions on the test set...")
    for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        text = row['text']
        true_label = row['label']

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        
        predicted_class_id = torch.argmax(logits, dim=1).item()
        
        predictions.append(predicted_class_id)
        true_labels.append(true_label)

    # --- 5. Calculateing and Display Metrics ---
    print("\n--- Evaluation Results ---")
    
    # Classification Report (Precision, Recall, F1-Score)
    report = classification_report(true_labels, predictions, target_names=['Not Abusive', 'Abusive'])
    print(report)

    # Confusion Matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Abusive', 'Abusive'])
    
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    
    # Save the plot
    output_fig_path = os.path.join(project_root, "evaluation_confusion_matrix.png")
    plt.savefig(output_fig_path)
    print(f"Confusion matrix saved to {output_fig_path}")
    plt.show()

if __name__ == '__main__':
    evaluate_model()