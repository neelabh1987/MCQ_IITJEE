import streamlit as st
import requests

# --- Fireworks.ai API Setup ---
FIREWORKS_API_KEY = st.secrets["FIREWORKS_API_KEY"]  # Store in Streamlit secrets
API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {FIREWORKS_API_KEY}",
    "Accept": "application/json",
    "Content-Type": "application/json",
}

# --- Streamlit UI ---
st.set_page_config(page_title="JEE MCQ Generator", page_icon="🧠")
st.title("🧠 JEE MCQ Generator (Powered by Mistral-7B)")
st.caption("Enter a concept to generate an exam-style MCQ.")

# --- Improved MCQ Generation Function ---
def generate_mcq(concept):
    PROMPT = f"""
    Create a JEE-level MCQ (Physics/Chemistry/Math) about: {concept}
    Rules:
    - Format: Clear question + 4 options labeled (a, b, c, d)
    - No explanations or answers
    - Example format:
    **Question:** What is the SI unit of force?
    **Options:**
    (a) Joule
    (b) Newton
    (c) Watt
    (d) Pascal
    """
    
    payload = {
        "model": "accounts/fireworks/models/mistral-7b-instruct-4k",
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.7,
        "max_tokens": 300,
        "top_p": 0.9,
    }
    
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()  # Raises HTTPError for bad responses
        
        data = response.json()
        
        # Check if 'choices' exists in response
        if "choices" not in data:
            st.error("Unexpected API response format. Please try again.")
            st.json(data)  # Debug: Show full response
            return None
            
        return data["choices"][0]["message"]["content"]
        
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# --- User Input ---
user_input = st.text_area(
    "✍️ Enter a JEE concept:", 
    placeholder="e.g., Projectile motion, Organic chemistry mechanisms...",
    height=100
)

if st.button("🎯 Generate MCQ", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a concept!")
    else:
        with st.spinner("Generating your MCQ..."):
            mcq = generate_mcq(user_input)
            
            if mcq:
                st.success("Here's your generated MCQ:")
                st.markdown(f"```\n{mcq}\n```")  # Display in monospace for clarity
                
                # Optional: Pretty formatting
                st.divider()
                st.markdown("**Formatted Version:**")
                st.write(mcq)  # Renders markdown

