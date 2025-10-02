import Main_classification
import classification_data
import streamlit as st
from Main_classification import render_sidebar
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import wikipedia
import re

# -------------------------
# Sidebar (persistent + context input)
# -------------------------
with st.sidebar:
    render_sidebar(
        Main_classification.training_text,
        Main_classification.training_pclass,
        Main_classification.training_sclass,
        Main_classification.accuracies
    )

    st.markdown("---")
    st.markdown("### Optional context passage")
    context_input = st.text_area("Paste your context here:", height=200)

context_input = context_input.strip() if context_input else ""

# -------------------------
# Chat panel
# -------------------------
st.markdown('<h1 style="font-size:70px">Welcome, User.</h1>', unsafe_allow_html=True)
st.markdown('<h1 style="font-size:30px">What\'s on today\'s agenda?</h1>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------
# Helper functions
# -------------------------
def split_sentences(text):
    return re.split(r'(?<=[.!?]) +', text)

def top_relevant_sentences(paragraph, question, n=3):
    sentences = split_sentences(paragraph)
    question_words = set(question.lower().split())
    scored = [(s, len(question_words & set(s.lower().split()))) for s in sentences]
    scored.sort(key=lambda x: x[1], reverse=True)
    return ' '.join([s for s, _ in scored[:n]]) if scored else paragraph

def clean_text(text):
    # Remove footnote markers like [a], [1], etc.
    return re.sub(r'\[\w+\]', '', text)

# -------------------------
# Load extractive QA model from Hugging Face repo
# -------------------------
# Replace with your actual HF repo ID
HF_REPO = "GryffinT/SQuAD.QA"

tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
model = AutoModelForQuestionAnswering.from_pretrained(HF_REPO)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

def answer_question(question, context):
    if not context.strip():
        return "No context provided."
    context_clean = clean_text(context)
    result = qa_pipeline(question=question, context=context_clean)
    return result.get("answer", "No answer found")

# -------------------------
# Chat input
# -------------------------
if prompt := st.chat_input("Ask Laurent anything."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. Classification
        classifications = Main_classification.pipeline.predict(prompt)

        # 2. Determine context
        fetch_context = context_input

        # 3. Auto-fetch Wikipedia page if context is empty
        if not fetch_context:
            try:
                page = wikipedia.page(prompt)
                # Use the first paragraph as default context
                first_para = page.content.split('\n')[0]
                fetch_context = first_para
            except Exception:
                fetch_context = "No context found for this question."

        # 4. Multi-sentence retrieval for completeness
        relevant_context = top_relevant_sentences(fetch_context, prompt, n=3)

        # 5. Get extractive answer
        answer = answer_question(prompt, relevant_context)

        # 6. Display classifications + answer
        response = f"The classifications are: {classifications}, and my answer is {answer}"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
