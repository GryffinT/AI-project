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

if prompt := st.chat_input("Ask me anything"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        classifications = Main_classification.pipeline.predict(statement)
        response = f"The classifications are: {classifications}."
    st.session_state.messages.append({"role": "assistant", "content": response})
      
      


