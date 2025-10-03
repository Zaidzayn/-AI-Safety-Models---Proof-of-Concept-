import streamlit as st
from src.orchestrators import EnglishOrchestrator, MultilingualOrchestrator

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Safety POC", layout="wide")

# --- MODEL LOADING (with switching logic) ---
@st.cache_resource
def load_orchestrator(model_choice):
    """Loads the selected SafetyOrchestrator and caches it."""
    try:
        if model_choice == "Multilingual (XLM-RoBERTa)":
            return MultilingualOrchestrator()
        else: # Default to English
            return EnglishOrchestrator()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please make sure all trained model files are in the 'models' directory and have the correct names.")
        return None

# --- SIDEBAR ---
st.sidebar.title("Configuration")
model_selection = st.sidebar.selectbox(
    "Choose the model to use:",
    ("English-Only (RoBERTa)", "Multilingual (XLM-RoBERTa)")
)

orchestrator = load_orchestrator(model_selection)

if st.sidebar.button("Clear Chat History"):
    history_key = f"messages_{model_selection}"
    st.session_state[history_key] = []
    if orchestrator:
        orchestrator.conversation_history.clear()
    st.rerun()

# --- MAIN APP LOGIC ---
st.title("🏆 AI Safety POC")
st.markdown(f"Currently running with the **{model_selection}** model.")

if orchestrator is None:
    st.stop()

# Initialize or get session state for conversation history
history_key = f"messages_{model_selection}"
if history_key not in st.session_state:
    st.session_state[history_key] = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = "streamlit_chat_combined_123"

# Display prior chat messages
for message in st.session_state[history_key]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT INPUT & PROCESSING ---
if prompt := st.chat_input("Enter your message..."):
    st.session_state[history_key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Analyzing..."):
        abuse_score = orchestrator.analyze_abuse(prompt)
        crisis_score = orchestrator.analyze_crisis(prompt)
        escalation_level = orchestrator.analyze_escalation(st.session_state.conversation_id, prompt)
        content_filter_status = orchestrator.analyze_content_filter(prompt, user_age=15, abuse_score=abuse_score)

        # Determine the final decision with priority
        final_decision = "✅ ALLOW MESSAGE"
        
        if crisis_score >= 1.0:
            final_decision = "🚨 URGENT: ALERT HUMAN INTERVENTION"
        elif crisis_score >= 0.8:
            final_decision = "⚠️ HIGH CONCERN: FLAG FOR REVIEW"
        elif content_filter_status != "ALLOWED":
            final_decision = f"❌ BLOCK MESSAGE ({content_filter_status})"
        elif abuse_score > 0.7:
            final_decision = "❌ BLOCK MESSAGE (High Abuse Score)"
        elif escalation_level == "HIGH":
            final_decision = "⚠️ FLAG CONVERSATION (Escalation Detected)"

    # Display assistant's response and analysis
    with st.chat_message("assistant"):
        response_content = f"**Decision:** {final_decision}"
        st.markdown(response_content)
        with st.expander("See Detailed Analysis"):
            st.metric("Abuse Score", f"{abuse_score:.3f}")
            st.metric("Crisis Score", f"{crisis_score:.2f}")
            st.metric("Escalation Level", escalation_level)
            st.metric("Content Filter (Age 15)", content_filter_status)
            st.info(f"**Model Used:** {model_selection}", icon="🤖")

    st.session_state[history_key].append({"role": "assistant", "content": response_content})