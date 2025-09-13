import Main_classification
import classification_data
import streamlit as st
from Main_classification import render_sidebar

# -------------------------
# Sidebar (persistent)
# -------------------------
render_sidebar(
    Main_classification.training_text,
    Main_classification.training_pclass,
    Main_classification.training_sclass,
    Main_classification.accuracies
)

# -------------------------
# Chat panel
# -------------------------

st.title("Welcome, User.")
st.write("What's on today's agenda?")

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
        response = f"The classifications are: {classifications}."
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

      


