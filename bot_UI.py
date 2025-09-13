import Main_classification
import classification_data
import streamlit as st
import random

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.title("Laurant.CA")
st.write ("Logistic-Regression Transformer Classifcation Algorithm")

random_message = data.keys()(math.rand(0,5))
if prompt := st.chat_input(random_message):
  with st.chat_message("user"):
    st.markdown(prompt)
  st.session_state.messages.append({"role": "user", "content": prompt})

