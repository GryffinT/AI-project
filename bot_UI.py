import re
import streamlit as st
import wikipedia
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import classification_data
from Main_generative import output
from Main_classification import render_sidebar
import Main_classification

# -------------------------
# Config
# -------------------------
HF_REPO = "GryffinT/SQuAD.QA"  # <-- your Hugging Face repo id


# -------------------------
# Sidebar: render sidebar + context input
# -------------------------
with st.sidebar:
    render_sidebar(
        Main_classification.training_text,
        Main_classification.training_pclass,
        Main_classification.training_sclass,
        Main_classification.accuracies
    )
    
# -------------------------
# Chat panel
# -------------------------

st.markdown('<h1 style="font-size:70px">Welcome, User.</h1>', unsafe_allow_html=True)
st.markdown('<h1 style="font-size:30px">What\'s on today\'s agenda?</h1>', unsafe_allow_html=True)
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask Laurent anything."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        classifications = Main_classification.pipeline.predict(prompt)
        generation = output(prompt, "")
        def wrap_text(text, words_per_line=7):
            words = text.split()
            lines = [' '.join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)]
            return '\n'.join(lines)
        
        wrapped_generation = wrap_text(generation, words_per_line=7)
        response = f"### Classifications\n```text\n{wrapped_generation}\n```"
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
