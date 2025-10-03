import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from src.escalation_model_def import EscalationClassifier
import os

class BaseOrchestrator:
    """A base class to hold shared methods for all orchestrators."""
    def __init__(self):
        self.conversation_history = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Orchestrator ready. Using device: {self.device}")

    def analyze_abuse(self, text):
        """Analyzes text for abuse using the loaded abuse model."""
        inputs = self.abuse_tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.abuse_model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1)
        abuse_score = probabilities[0][1].item()
        return abuse_score

    def analyze_content_filter(self, text, user_age, abuse_score):
        """Applies a tiered, rule-based content filter based on user age and abuse score."""
        text_lower = text.lower()
        child_strict_keywords = ['damn', 'hell'] 

        if user_age < 13:
            if abuse_score > 0.4:
                return "BLOCKED_CHILD (Moderate Abuse)"
            for word in child_strict_keywords:
                if word in text_lower:
                    return "BLOCKED_CHILD (Strict Keyword)"
            return "ALLOWED"
        elif user_age < 18:
            if abuse_score > 0.7:
                return "BLOCKED_TEEN (High Abuse)"
            return "ALLOWED"
        else:
            if abuse_score > 0.9:
                return "BLOCKED_ADULT (Severe Abuse)"
            return "ALLOWED"

class EnglishOrchestrator(BaseOrchestrator):
    """Orchestrator for English-only models, with ML models for Abuse and Escalation."""
    def __init__(self, 
                 abuse_model_folder="abuse-detector-roberta-lora",
                 escalation_model_folder="escalation-detector-lstm"):
        super().__init__()
        print("Initializing ENGLISH-ONLY Safety Orchestrator...")
        
        script_dir = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(script_dir, ".."))
        
        # --- Load Abuse Model ---
        abuse_model_path = os.path.normpath(os.path.join(project_root, "models", abuse_model_folder))
        base_model_name = "roberta-base"
        self.abuse_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        abuse_base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
        self.abuse_model = PeftModel.from_pretrained(abuse_base_model, abuse_model_path).to(self.device)
        self.abuse_model.eval()

        # --- Load Escalation Model ---
        st_model_name = 'all-MiniLM-L6-v2'
        self.st_model = SentenceTransformer(st_model_name, device=self.device)
        
        escalation_model_path = os.path.normpath(os.path.join(project_root, "models", escalation_model_folder, "escalation_lstm.pth"))
        input_size = self.st_model.get_sentence_embedding_dimension()
        self.escalation_model = EscalationClassifier(input_size=input_size, hidden_size=128).to(self.device)
        self.escalation_model.load_state_dict(torch.load(escalation_model_path, map_location=self.device))
        self.escalation_model.eval()

    def analyze_crisis(self, text):
        """Analyzes text for crisis content using a tiered keyword system."""
        text_lower = text.lower()
        urgent_keywords = ["kill myself", "end my life", "suicide"]
        for keyword in urgent_keywords:
            if keyword in text_lower:
                return 1.0

        high_concern_keywords = ["want to die", "can't go on", "self-harm", "no reason to live"]
        for keyword in high_concern_keywords:
            if keyword in text_lower:
                return 0.8

        return 0.0

    def analyze_escalation(self, conversation_id, current_message_text):
        """Analyzes conversation history for escalation using the LSTM model."""
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []
        
        history = self.conversation_history[conversation_id]
        history.append(current_message_text)
        self.conversation_history[conversation_id] = history[-5:]
        
        if len(self.conversation_history[conversation_id]) < 3:
            return "NORMAL"
            
        sequence_to_analyze = self.conversation_history[conversation_id]
        
        with torch.no_grad():
            embeddings = self.st_model.encode(sequence_to_analyze, convert_to_tensor=True, device=self.device)
            embeddings = embeddings.unsqueeze(0)
            logits = self.escalation_model(embeddings)
            prediction = torch.sigmoid(logits).item()

        if prediction > 0.5:
            return "HIGH"
        return "NORMAL"

class MultilingualOrchestrator(BaseOrchestrator):
    """Orchestrator for the multilingual model, with placeholder crisis/escalation."""
    def __init__(self, model_folder_name="abuse-detector-xlm-roberta-lora"):
        super().__init__()
        print("Initializing MULTILINGUAL Safety Orchestrator...")
        
        script_dir = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(script_dir, ".."))
        model_path = os.path.normpath(os.path.join(project_root, "models", model_folder_name))
        
        base_model_name = "xlm-roberta-base"
        self.abuse_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
        self.abuse_model = PeftModel.from_pretrained(base_model, model_path).to(self.device)
        self.abuse_model.eval()

    def analyze_crisis(self, text):
        """Analyzes text for crisis content with multilingual keywords."""
        text_lower = text.lower()
        urgent_keywords = ["kill myself", "end my life", "suicide", "muje sucide karni hai"]
        for keyword in urgent_keywords:
            if keyword in text_lower:
                return 1.0

        high_concern_keywords = [
            "want to die", "can't go on", "self-harm", "no reason to live",
            "main marna chahta hoon", "jeene ka man nahi karta"
        ]
        for keyword in high_concern_keywords:
            if keyword in text_lower:
                return 0.8

        return 0.0
    
    def analyze_escalation(self, conversation_id, current_message_text):
        """Placeholder for escalation detection (uses abuse score as a proxy)."""
        abuse_score = self.analyze_abuse(current_message_text)
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []
        history = self.conversation_history[conversation_id]
        history.append(abuse_score)
        self.conversation_history[conversation_id] = history[-5:]
        high_score_count = sum(1 for score in history if score > 0.75)
        if len(history) >= 3 and high_score_count >= 3:
            return "HIGH"
        return "NORMAL"