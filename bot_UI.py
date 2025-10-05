import re
import streamlit as st
import wikipedia
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import Main_classification
import classification_data
from Main_classification import render_sidebar

# -------------------------
# Config
# -------------------------
HF_REPO = "GryffinT/SQuAD.QA"  # <-- your Hugging Face repo id
CHUNK_MAX_WORDS = 700         # size of each chunk in words
CHUNK_OVERLAP = 120           # overlap between chunks in words
SCORE_THRESHOLD = 0.0         # include answers with score >= this
MAX_AGG_ANSWERS = 5          # max number of distinct answers to include
MAX_FINAL_CHARS = 1000       # truncate final aggregated answer to this many chars

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
    st.markdown("---")
    st.markdown("### Optional context passage (sidebar)")
    context_input = st.text_area("Paste your context here (optional):", height=220)
    st.markdown("---")
    debug_mode = st.checkbox("Show debug info (context, chunks)", value=False)

context_input = context_input.strip() if context_input else ""

# Response

#RESPONSE STUFF
