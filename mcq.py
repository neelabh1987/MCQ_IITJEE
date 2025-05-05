import streamlit as st
import requests

# --- Updated Fireworks.ai API Configuration ---
FIREWORKS_API_KEY = st.secrets["FIREWORKS_API_KEY"]
API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"  # Correct endpoint
MODEL_NAME = "accounts/fireworks/models/mixtral-8x7b-instruct"  # Recommended model

# --- Streamlit App ---
st.set_page_config(page_title="JEE MCQ Generator", page_icon="🧠")
st.title("🧠 JEE MCQ Generator v2.0")

def generate_mcq(topic):
    """Generate MCQ using Fireworks.ai's chat API"""
    messages = [
        {
            "role": "system",
            "content": "You are an expert JEE exam creator. Generate 1 MCQ with 4 options (a-d). No explanations."
        },
        {
            "role": "user",
            "content": f"Create a JEE-level MCQ about: {topic}\n\nFormat:\nQuestion: [Your question]\na) Option 1\nb) Option 2\nc) Option 3\nd) Option 4"
        }
    ]
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 256,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {FIREWORKS_API_KEY}"},
            json=payload,
            timeout=30  # Added timeout
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    except requests.exceptions.HTTPError as err:
        st.error(f"HTTP Error: {err}\n\nAPI Response: {response.text}")
    except Exception as err:
        st.error(f"Unexpected error: {err}")

# --- UI Components ---
with st.form("mcq_form"):
    concept = st.text_area("Enter JEE concept:", height=100)
    submitted = st.form_submit_button("Generate MCQ")
    
    if submitted and concept:
        with st.spinner("Generating..."):
            result = generate_mcq(concept)
            if result:
                st.success("### Generated MCQ")
                st.markdown(result)

