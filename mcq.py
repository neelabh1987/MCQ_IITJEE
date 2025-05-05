import streamlit as st
import requests

# --- Fireworks.ai API Setup ---
FIREWORKS_API_KEY = "fw_3ZT4V28a35wZKEvjwdp216x3"  
API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {FIREWORKS_API_KEY}",
    "Content-Type": "application/json",
}

# --- Streamlit UI ---
st.set_page_config(page_title="JEE MCQ Generator", page_icon="🧠")
st.title("🧠 JEE MCQ Generator (Powered by Mistral-7B)")
st.caption("Enter a concept to generate an exam-style MCQ.")

# --- MCQ Generation Function ---
def generate_mcq(concept):
    PROMPT = f"""
    Create a JEE-level MCQ (Physics/Chemistry/Math) about: {concept}
    - Format: Clear question + 4 options (a, b, c, d)
    - No explanations or answers
    - Strictly follow this template:
    **Question:** [Your question here]
    **Options:**
    (a) Option 1
    (b) Option 2
    (c) Option 3
    (d) Option 4
    """
    
    payload = {
        "model": "accounts/fireworks/models/mistral-7b-instruct-4k",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 300,
        "temperature": 0.7,
    }
    
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.json()["choices"][0]["message"]["content"]

# --- User Input ---
user_input = st.text_area("✍️ Enter a JEE concept:", placeholder="e.g., Projectile motion, Organic chemistry mechanisms...")

if st.button("🎯 Generate MCQ"):
    if not user_input.strip():
        st.warning("Please enter a concept!")
    else:
        with st.spinner("Generating..."):
            try:
                mcq = generate_mcq(user_input)
                st.markdown(f"**Generated MCQ:**\n\n{mcq}")
            except Exception as e:
                st.error(f"Error: {str(e)}. Try again later.")


# Footer
st.markdown("<div class='footer'>Model: google/flan-t5-small • Powered by HuggingFace + Streamlit</div>", unsafe_allow_html=True)

