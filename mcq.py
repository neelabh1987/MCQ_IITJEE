import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"  # Add this line

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
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

# Load the Llama-3-8B-Instruct model
@st.cache_resource(show_spinner="🚀 Loading Llama-3.2-3B-Instruct model...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        token=st.secrets["HUGGINGFACE_TOKEN"]  # Add this line
    )
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        token=st.secrets["HUGGINGFACE_TOKEN"],  # Add this line
        torch_dtype=torch.float16,  # Optional: reduces memory usage
        device_map="auto"
    )
    return model, tokenizer

# Function to generate MCQ
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

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    # Format the prompt for Llama 3 instruct model
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Generate the response
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=256)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the output
    output = output.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    output = re.sub(r'Subject:\s*\w+\s*', '', output)
    output = output.replace('\\n', '\n').replace('\n\n', '\n').strip()

    # Extracting question and options
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
st.markdown("<div class='footer'>Model: meta-llama/Llama-3.2-3B-Instruct • Powered by HuggingFace + Streamlit</div>", unsafe_allow_html=True)

