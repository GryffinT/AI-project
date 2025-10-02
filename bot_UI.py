import Main_classification
import classification_data
import streamlit as st
from Main_classification import render_sidebar
from Main_generative import output
import wikipedia

# -------------------------
# Sidebar (persistent)
# -------------------------
render_sidebar(
    Main_classification.training_text,
    Main_classification.training_pclass,
    Main_classification.training_sclass,
    Main_classification.accuracies
    context_input = st.text_area("Paste your context here (optional):", height=200)
    context = context_input.strip()
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
        # Classification
        classifications = Main_classification.pipeline.predict(prompt)

        # Auto-fetch from Wikipedia if empty
        if not context:
            try:
                page = wikipedia.page(prompt)
                context = page.content[:1000]  # first 1000 chars as context
            except Exception:
                context = "No context found for this question."

        # Extractive answer
        answer = output(prompt, context)

        # Full response
        response = f"The classifications are: {classifications}, and my answer is {answer}"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

