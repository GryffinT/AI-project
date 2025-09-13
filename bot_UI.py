import Main_classification
import classification_data
import streamlit as st
import random

if "messages" not in st.session_state:
  st.session_state.messages = []

for message in st.session_state.message:
  with st.chat_message(message=["role"]):
    st.markdown(message["content"])

st.title("Laurant.CA")
st.write ("Logistic-Regression Transformer Classifcation Algorithm")

random_message = data.keys()(math.rand(0,5))
if prompt := st.chat_input(random_message):
  with st.chat_message("user"):
    st.markdown(prompt)
  st.session_state.messages.append("role": "user", "content": prompt)

