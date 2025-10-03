import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import os

#  1. Configuration 
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "st_model": "all-MiniLM-L6-v2", 
    "data_path": "data/processed/escalation_dataset.csv",
    "output_dir": "models/escalation-detector-lstm",
    "lstm_hidden_size": 128,
    "epochs": 5, # we should train for more epochs in practice to get better results
    "batch_size": 32,
    "learning_rate": 1e-4,
}

#  Custom PyTorch Dataset 
class EscalationDataset(Dataset):
    """Custom dataset for loading and encoding conversation sequences."""
    def __init__(self, dataframe, st_model):
        self.df = dataframe
        self.st_model = st_model
        # The ST model produces embeddings of size 384 for 'all-MiniLM-L6-v2'
        self.embedding_dim = self.st_model.get_sentence_embedding_dimension()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sequence_text = row['text_sequence'].split(' [SEP] ')
        label = torch.tensor(row['label'], dtype=torch.float32)

        # Encodeing the sequence of sentences into a sequence of embeddings
        embeddings = self.st_model.encode(sequence_text, convert_to_tensor=True, device=CONFIG["device"])
        
        return embeddings, label

# PyTorch Model Definition 
class EscalationClassifier(nn.Module):
    """LSTM model to classify a sequence of sentence embeddings."""
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(EscalationClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 1) # Output a single logit for binary classification

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, (hidden, _) = self.lstm(x)
        # We use the last hidden state of the LSTM for classification
        final_hidden_state = hidden.squeeze(0)
        logits = self.classifier(final_hidden_state)
        return logits.squeeze(-1)

# --- Main Execution Block ---
if __name__ == '__main__':
    print(f"Using device: {CONFIG['device']}")
    
    # --- 4. Load Data ---
    df = pd.read_csv(CONFIG['data_path'])
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    print("Loading Sentence-Transformer model...")
    sentence_transformer_model = SentenceTransformer(CONFIG['st_model'], device=CONFIG['device'])

    train_dataset = EscalationDataset(train_df, sentence_transformer_model)
    val_dataset = EscalationDataset(val_df, sentence_transformer_model)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])

    # --- 5. Initialize Model, Loss, and Optimizer ---
    model = EscalationClassifier(input_size=train_dataset.embedding_dim, hidden_size=CONFIG['lstm_hidden_size']).to(CONFIG['device'])
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    
    # --- 6. Training Loop ---
    print("\nStarting model training...")
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        
        for embeddings, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}"):
            embeddings, labels = embeddings.to(CONFIG['device']), labels.to(CONFIG['device'])
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"\nEpoch {epoch+1} - Training Loss: {total_loss/len(train_loader):.4f}")

        # --- 7. Evaluation Loop ---
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(CONFIG['device']), labels.to(CONFIG['device'])
                outputs = model(embeddings)
                
                preds = torch.sigmoid(outputs) > 0.5
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        
        print(f"Validation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")

    #  Save the Model ---
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    model_save_path = os.path.join(CONFIG['output_dir'], 'escalation_lstm.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Training complete. Model saved to {model_save_path}")