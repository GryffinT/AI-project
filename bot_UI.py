import Main_classification
import classification_data
import streamlit as st
from Main_classification import render_sidebar
from Main_generative import output
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

# Strip whitespace
context_input = context_input.strip() if context_input else ""

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

# -------------------------
# Helper: split text into sentences
# -------------------------
def split_sentences(text):
    return re.split(r'(?<=[.!?]) +', text)

# -------------------------
# Helper: retrieve top relevant sentences
# -------------------------
def top_relevant_sentences(paragraph, question, n=3):
    sentences = split_sentences(paragraph)
    question_words = set(question.lower().split())
    scored = [(s, len(question_words & set(s.lower().split()))) for s in sentences]
    scored.sort(key=lambda x: x[1], reverse=True)  # sort by overlap
    return ' '.join([s for s, _ in scored[:n]]) if scored else paragraph

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

        # 3. Auto-fetch entire Wikipedia page if context is empty
        if not fetch_context:
            try:
                page = wikipedia.page(prompt)
                # Use the first paragraph as summary
                first_para = page.content.split('\n')[0]
                fetch_context = first_para
            except Exception:
                fetch_context = "No context found for this question."

        # 4. Retrieve top relevant sentences (multi-sentence)
        relevant_context = top_relevant_sentences(fetch_context, prompt, n=3)

        # 5. Get answer from the QA model
        answer = output(prompt, relevant_context)

        # 6. Keep first 2-3 sentences from the model output for completeness
        answer_sentences = split_sentences(answer)
        final_answer = ' '.join(answer_sentences[:3]) if answer_sentences else answer

        # 7. Full response
        response = f"The classifications are: {classifications}, and my answer is {final_answer}"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

