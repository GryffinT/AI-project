import Main_classification
import classification_data
import streamlit as st
from Main_classification import render_sidebar
from Main_generative import output
import wikipedia

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
context = context_input.strip() if context_input else ""

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
# Chat input
# -------------------------
if prompt := st.chat_input("Ask Laurent anything."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. Classification
        classifications = Main_classification.pipeline.predict(prompt)

        # 2. Auto-fetch context from Wikipedia if empty
        fetch_context = context
        if not fetch_context:
            try:
                page = wikipedia.page(prompt)
                fetch_context = page.content[1000:]  # first 1000 chars
            except Exception:
                fetch_context = "No context found for this question."

        # 3. Extractive QA
        answer = output(prompt, fetch_context)

        # 4. Full response
        response = f"{answer}"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

