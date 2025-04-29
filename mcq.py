#!pip install streamlit
#!pip install auto-gptq transformers accelerate

import streamlit as st
from transformers import AutoTokenizer, pipeline, TextStreamer
from auto_gptq import AutoGPTQForCausalLM
import re

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

@st.cache_resource(show_spinner="🚀 Loading Mistral-7B GPTQ model...")
def load_model():
    model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    model = AutoGPTQForCausalLM.from_quantized(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        revision="main",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    return model, tokenizer

model, tokenizer = load_model()

# Streamer + generator
streamer = TextStreamer(tokenizer)
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    do_sample=True,
    temperature=0.3,
    return_full_text=False,
    top_k=40,
    top_p=0.85
)

def generate_mcq(user_prompt):
    system_prompt = "You are an expert JEE MCQ creator for Physics, Chemistry, and Mathematics."
    user_message = f"""Create one MCQ based on this concept: {user_prompt}

Rules:
- Identify subject (Physics, Chemistry, Math)
- Create one question based on it
- Write four options (a), (b), (c), (d)
- No explanations, no answers, no extra text
- Clear, exam-style MCQ
"""

    prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_message}\n<|assistant|>\n"
    output = generator(prompt)[0]['generated_text'].strip()

    output = re.sub(r'Subject:\s*\w+\s*', '', output)
    output = output.replace('\\n', '\n').replace('\n\n', '\n').strip()

    question_match = re.search(r'(Question[:\s]*)?(.*?)(Options:|\(a\)|\(A\))', output, re.DOTALL)
    question_text = question_match.group(2).strip() if question_match else ""

    options = re.findall(r'\([a-dA-D]\)\s*(.*?)\s*(?=\([a-dA-D]\)|$)', output, re.DOTALL)

    if question_text and len(options) >= 4:
        formatted_lines = [f"**Question:** {question_text}", "\n**Options:**"]
        for i, opt in enumerate(options[:4]):
            formatted_lines.append(f"- ({chr(97+i)}) {opt.strip()}")
        return '\n'.join(formatted_lines)
    else:
        return f"⚠️ Could not extract proper question and options.\n\n**Raw Output:**\n```\n{output}\n```"

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
st.markdown("<div class='footer'>Model: TheBloke/Mistral-7B-Instruct-v0.2-GPTQ • Powered by HuggingFace + Streamlit</div>", unsafe_allow_html=True)
