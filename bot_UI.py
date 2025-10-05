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
    
# Response

