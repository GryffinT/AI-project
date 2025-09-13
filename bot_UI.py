import main_classification
import classification_data
import streamlit as st
from main_classification import render_sidebar

# -------------------------
# Sidebar (persistent)
# -------------------------
render_sidebar(
    classification_data.training_text,
    classification_data.training_pclass,
    classification_data.training_sclass,
    classification_data.accuracies
)

# -------------------------
# Chat panel
# -------------------------
st.header("Chat with your LLM")
user_input = st.text_input("Enter a message:")

if st.button("Send") and user_input:
    response = "This is a dummy LLM response"  # Replace with your LLM function
    st.write(f"LLM: {response}")

st.title("Laurant.CA")
st.write("Logistic-Regression Transformer Classification Algorithm")

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

      


