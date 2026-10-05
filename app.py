import streamlit as st
import whisper
import requests
import tempfile
import os
import json
from datetime import datetime

st.set_page_config(
    page_title="Properly", 
    layout="wide"
)

DB_FILE = "episodic_memories.json"

# --- Storage Layer ---
def load_logs():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_log(entry_text):
    logs = load_logs()
    logs.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "raw_text": entry_text
    })
    with open(DB_FILE, "w") as f:
        json.dump(logs, f, indent=2)

# --- Model Loaders ---
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

whisper_model = load_whisper()

def query_gemma(prompt: str, system_prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "gemma2:2b",
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.9}
    }
    try:
        response = requests.post(url, json=payload, timeout=60)
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"Error connecting to Ollama: {e}"

# --- Prompt Engine ---
PROMPT_CONVO_AMMO = """
You are a conversational assistant helping an analytical person who frequently gets told "you don't talk properly" or who draws a complete blank during long calls and chats.
The user has logged raw, unpolished daily audio notes.

Extract actionable, copy-pasteable conversational fuel calibrated for the selected recipient type.
Do NOT use any emojis in your response.

Format strictly as:
### 1. Three Story Drops (To answer "What's up with you lately?")
Turn dry facts into casual, human micro-stories (1-2 sentences each). Include the small textures (food, delays, weird thoughts, small wins).

### 2. Three Return Questions (To keep the spotlight off you)
Natural, curious follow-up questions that make the other person talk.

### 3. Banter / Relatable Hook (When the conversation dies)
A casual observation, grievance, or shared absurdity from the notes to bring up.
"""

PROMPT_DECODER = """
You are an interpersonal translator between analytical, direct thinkers and relational communicators.
The user received a message from someone in their life.
Do NOT use any emojis in your response.

1. Subtext / Underlying Need: State in 1 sentence what the sender is actually seeking (e.g. empathy, reassurance, casual banter, or concrete planning).
2. Recommended Reply: Provide a warm, natural reply that addresses their emotional cadence without sounding fake or corporate.
"""

# --- UI Layout ---
st.title("Properly")
st.caption("A local-first communication bridge for turning raw thoughts into conversational fuel.")

tab_intake, tab_reply = st.tabs(["Daily Debrief & Convo Fuel", "Message Decoder"])

with tab_intake:
    col_left, col_right = st.columns([1, 1.2], gap="large")
    
    with col_left:
        st.subheader("1. 60-Second Daily Intake")
        st.write("Rant for 30–60 seconds about whatever happened today: code hurdles, food, delays, things that annoyed you, or random observations.")
        
        audio_data = st.audio_input("Record Daily Audio", key="daily_audio")
        
        if audio_data is not None:
            if st.button("Transcribe & Save to Memory", type="primary"):
                with st.spinner("Transcribing locally via Whisper..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_data.getvalue())
                        tmp_path = tmp.name

                    result = whisper_model.transcribe(tmp_path)
                    transcription = result["text"].strip()
                    os.remove(tmp_path)

                save_log(transcription)
                st.success("Saved to local memory.")
                st.info(f"Logged: \"{transcription}\"")

        logs = load_logs()
        st.markdown("---")
        st.caption(f"Stored Memory Vault: {len(logs)} entries")
        if logs:
            with st.expander("View Recent Logs"):
                for entry in reversed(logs[-3:]):
                    st.caption(f"[{entry['timestamp']}] — {entry['raw_text']}")

    with col_right:
        st.subheader("2. Generate Convo Fuel")
        st.write("Mid-conversation, on a call, or about to text? Generate material so conversations don't die after two minutes.")
        
        recipient_profile = st.selectbox(
            "Who are you talking to?", 
            [
                "Relational Friend (Needs context, stories, and warm banter)",
                "Parent / Family (Needs life updates, health, and wellbeing checks)",
                "Casual Friend / Peer (Banter, shared gripes, and mutual interests)"
            ]
        )
        days_lookback = st.slider("Pull from how many recent days?", 1, 7, value=min(3, max(1, len(logs))))
        
        if st.button("Generate Fuel & Stories", type="primary"):
            if not logs:
                st.warning("No entries in your memory vault yet. Record a quick voice log on the left first.")
            else:
                recent = logs[-days_lookback:]
                context = "\n".join([f"- [{x['timestamp']}]: {x['raw_text']}" for x in recent])
                
                with st.spinner("Extracting conversational ammo with Gemma 2..."):
                    ammo = query_gemma(
                        f"Recipient: {recipient_profile}\nRecent Notes:\n{context}",
                        PROMPT_CONVO_AMMO
                    )
                st.markdown(ammo)

with tab_reply:
    st.subheader("Message Decoder & Tone Calibrator")
    st.write("Paste a message you received that you aren't sure how to answer without sounding blunt, dry, or dismissive.")
    
    received_msg = st.text_area("Message you received:", placeholder="e.g. 'You've been super quiet lately. Is everything okay?' or 'Had the worst day at work today.'")
    
    if st.button("Decode Message", type="primary"):
        if not received_msg:
            st.warning("Paste a message first.")
        else:
            with st.spinner("Analyzing intent with Gemma 2..."):
                decoded = query_gemma(received_msg, PROMPT_DECODER)
            st.markdown(decoded)
