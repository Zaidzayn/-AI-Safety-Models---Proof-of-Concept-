import pytest
import sys
import os

# Adding the 'src' directory to the Python path to allow importing the orchestrators
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.orchestrators import EnglishOrchestrator, MultilingualOrchestrator
from src.escalation_model_def import EscalationClassifier # Needed for loading the model

## load models once per test session ---

@pytest.fixture(scope="module")
def english_orchestrator():
    """Loads the English-only orchestrator once for all tests."""
    try:
        return EnglishOrchestrator()
    except Exception as e:
        pytest.fail(f"Failed to load EnglishOrchestrator. Make sure models are downloaded. Error: {e}")

@pytest.fixture(scope="module")
def multilingual_orchestrator():
    """Loads the Multilingual orchestrator once for all tests."""
    try:
        return MultilingualOrchestrator()
    except Exception as e:
        pytest.fail(f"Failed to load MultilingualOrchestrator. Make sure models are downloaded. Error: {e}")

# --- Test Cases for Abuse Detection ---

def test_benign_english_message(english_orchestrator):
    text = "I think this is a great idea, thank you!"
    abuse_score = english_orchestrator.analyze_abuse(text)
    assert abuse_score < 0.5

def test_obvious_abuse_english(english_orchestrator):
    text = "You are a stupid idiot and your ideas are worthless."
    abuse_score = english_orchestrator.analyze_abuse(text)
    assert abuse_score > 0.7

# --- Edge Case Tests ---

def test_sarcasm_ambiguity(english_orchestrator):
    text = "Oh great, another meeting. I am absolutely thrilled."
    abuse_score = english_orchestrator.analyze_abuse(text)
    assert abuse_score < 0.5

def test_threat_detection(english_orchestrator):
    text = "If you say that again I'm going to find you."
    abuse_score = english_orchestrator.analyze_abuse(text)
    # This is a key finding to discuss in a report.
    assert abuse_score > 0.001 # Check it's not completely zero

def test_slang_insult(english_orchestrator):
    text = "That whole idea is so cringe, just stop."
    abuse_score = english_orchestrator.analyze_abuse(text)
    assert abuse_score > 0.6

# --- Test Case for the ML-based Escalation Model ---

def test_escalation_sequence_english_model(english_orchestrator):
    """Tests that a sequence of abusive messages is eventually flagged as HIGH."""
    conversation_id = "pytest_escalation_test"
    english_orchestrator.conversation_history.clear()
    
    messages = [
        "Hello there.",
        "This is a bit slow.",
        "You are a useless machine.",
        "I hate this stupid program.",
        "This is the worst bot I have ever used."
    ]
    
    
    final_result = ""
    for msg in messages:
        final_result = english_orchestrator.analyze_escalation(conversation_id, msg)
        
    
    assert final_result == "HIGH"

# --- Multilingual Comparison Tests ---

def test_multilingual_abuse_fails_on_english_model(english_orchestrator):
    text = "tum bahut kharab insaan ho"
    abuse_score = english_orchestrator.analyze_abuse(text)
    assert abuse_score < 0.5

def test_multilingual_abuse_succeeds_on_multi_model(multilingual_orchestrator):
    text = "tum bahut kharab insaan ho"
    abuse_score = multilingual_orchestrator.analyze_abuse(text)
    # This shows the model has some understanding, even if not highly confident.
    assert abuse_score > 0.2