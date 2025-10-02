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
# Helper: retrieve most relevant sentences
# -------------------------
def retrieve_relevant_sentences(text, question, max_sentences=5):
    sentences = split_sentences(text)
    question_words = set(question.lower().split())
    # Keep sentences containing at least one question word
    relevant = [s for s in sentences if question_words & set(s.lower().split())]
    return ' '.join(relevant[:max_sentences]) if relevant else ' '.join(sentences[:3])  # fallback to first 3 sentences

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
                fetch_context = page.content
            except Exception:
                fetch_context = "No context found for this question."

        # 4. Retrieve relevant sentences from the context
        relevant_context = retrieve_relevant_sentences(fetch_context, prompt, max_sentences=5)

        # 5. Get answer from the QA model
        answer = output(prompt, relevant_context)

        # 6. Pick first sentence to keep answer concise
        answer_sentences = split_sentences(answer)
        final_answer = answer_sentences[0] if answer_sentences else answer

        # 7. Full response
        response = f"The classifications are: {classifications}, and my answer is {final_answer}"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

