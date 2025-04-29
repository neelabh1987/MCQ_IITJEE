#!pip install streamlit
#!pip install auto-gptq transformers accelerate

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Page config
st.set_page_config(
    page_title="JEE MCQ Generator",
    page_icon="🧠",
    layout="centered"
)

# Custom CSS styling
st.markdown("""
    <style>
        .main {background-color: #f4f6fa;}
        .mcq-box {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 1rem;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            color: #4a4a4a;
            font-weight: 700;
            font-size: 2rem;
        }
        .option {
            margin-left: 15px;
        }
        .footer {
            font-size: 0.9rem;
            color: #888;
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown("<div class='title'>🧠 JEE MCQ Generator</div>", unsafe_allow_html=True)
st.markdown("Get exam-style MCQs instantly based on your concept prompt.")

@st.cache_resource(show_spinner="🚀 Loading flan-t5-small...")
def load_model():
    model_id = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

generator = load_model()

def generate_mcq(prompt):
    formatted_prompt = f"""Generate one JEE multiple-choice question based on the concept: "{prompt}"

Rules:
- Clearly state the subject (Physics, Chemistry, or Math) only if needed.
- Provide one question only.
- Write four options: (a), (b), (c), (d)
- Do NOT include explanation or answer
- Output must be concise and exam-style.
"""
    result = generator(formatted_prompt, max_new_tokens=256, temperature=0.7)[0]['generated_text']
    return result.strip()

# Input section
user_input = st.text_area("✍️ Enter concept prompt:", height=100, placeholder="e.g., Projectile motion and radius of curvature...")

if st.button("🎯 Generate MCQ"):
    if user_input.strip():
        with st.spinner("Generating MCQ..."):
            mcq = generate_mcq(user_input)
        st.markdown(f"<div class='mcq-box'>{mcq}</div>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a valid prompt to generate a question.")

# Footer
st.markdown("<div class='footer'>Model: google/flan-t5-small • Powered by HuggingFace + Streamlit</div>", unsafe_allow_html=True)

