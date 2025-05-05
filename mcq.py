import streamlit as st
import requests

# --- Fireworks.ai API Setup (Updated) ---
FIREWORKS_API_KEY = st.secrets["FIREWORKS_API_KEY"]  # Store in Streamlit secrets
API_URL = "https://api.fireworks.ai/inference/v1/completions"  # Updated endpoint
MODEL_NAME = "accounts/fireworks/models/mixtral-8x7b-instruct"  # Better alternative

# --- Streamlit UI ---
st.set_page_config(page_title="JEE MCQ Generator", page_icon="🧠")
st.title("🧠 JEE MCQ Generator (Powered by Fireworks.ai)")

# --- Fixed Generation Function ---
def generate_mcq(concept):
    PROMPT = f"""Generate a JEE Advanced MCQ with these rules:
1. Subject: Physics/Chemistry/Math
2. Concept: {concept}
3. Format:
**Question:** [Clear question]
**Options:**
a) [Option 1]
b) [Option 2]
c) [Option 3]
d) [Option 4]
4. No explanations or answers"""
    
    payload = {
        "model": MODEL_NAME,
        "prompt": PROMPT,  # Changed from 'messages' to 'prompt'
        "max_tokens": 300,
        "temperature": 0.3  # Lower for more factual responses
    }
    
    try:
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {FIREWORKS_API_KEY}"},
            json=payload
        )
        response.raise_for_status()
        return response.json()["choices"][0]["text"]
    
    except Exception as e:
        st.error(f"🚨 Error: {str(e)}")
        st.json(response.json()) if response else None  # Debug
        return None

# --- User Interface ---
user_input = st.text_area("Enter JEE concept:", height=100)
if st.button("Generate MCQ"):
    if user_input:
        with st.spinner("Generating..."):
            result = generate_mcq(user_input)
            st.markdown(result if result else "Failed to generate")

