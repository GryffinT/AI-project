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
# Helper: chunk text for long pages
# -------------------------
def chunk_text(text, chunk_size=300):
    """Split text into overlapping chunks of ~chunk_size words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# Helper: extract first sentence
def first_sentence(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences[0] if sentences else text

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

        # 4. Split context into smaller chunks
        chunks = chunk_text(fetch_context, chunk_size=300)

        # 5. Get answers from all chunks
        answers = []
        for chunk in chunks:
            ans = output(prompt, chunk)
            if ans != "No answer found":
                # Keep only the first sentence of each chunk's answer
                ans = first_sentence(ans)
                answers.append(ans)

        # 6. Pick the longest/best answer
        final_answer = max(answers, key=len) if answers else "No answer found"

        # 7. Full response
        response = f"The classifications are: {classifications}, and my answer is {final_answer}"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
