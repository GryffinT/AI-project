import Main_classification
import classification_data
import streamlit as st
import random
    
st.title("Laurant.CA")
st.write ("Logistic-Regression Transformer Classifcation Algorithm")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask Laurent anything."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        classifications = Main_classification.pipeline.predict(prompt)
        response = f"The classifications are: {classifications}."
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
      
      


