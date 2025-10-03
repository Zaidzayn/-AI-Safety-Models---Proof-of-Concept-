import torch.nn as nn

class EscalationClassifier(nn.Module):
    """LSTM model to classify a sequence of sentence embeddings."""
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(EscalationClassifier, self).__init__()
        # LSTM layer processes the sequence of embeddings
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # A final linear layer for binary classification
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        _, (hidden, _) = self.lstm(x)
        # We use the last hidden state from the LSTM for the final classification
        final_hidden_state = hidden.squeeze(0)
        logits = self.classifier(final_hidden_state)
        return logits.squeeze(-1)