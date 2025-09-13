import Main_classification
import classification_data
import streamlit as st
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# Initialize session state at the top
classification_data.init_session_state(training_text, training_pclass, training_sclass)

# Now you can safely use st.session_state.le_p_classes, etc.
with st.sidebar:

    fig, axes = plt.subplots(1, 2, figsize=(14,6))

    # Primary classifier scatter
    axes[0].scatter(st.session_state.X_pca[:,0], st.session_state.X_pca[:,1],
                    c=st.session_state.p_labels_encoded, cmap='tab10', alpha=0.7, edgecolor='k')
    axes[0].set_title("Primary Classifier")
    axes[0].legend(handles=[
        Line2D([0], [0], marker='o', color='w', label=cls,
               markerfacecolor=plt.cm.tab10(i / len(st.session_state.le_p_classes)), markersize=10)
        for i, cls in enumerate(st.session_state.le_p_classes)
    ], title="Classes")

    # Secondary classifier scatter
    axes[1].scatter(st.session_state.X_pca[:,0], st.session_state.X_pca[:,1],
                    c=st.session_state.s_labels_encoded, cmap='tab20', alpha=0.7, edgecolor='k')
    axes[1].set_title("Secondary Classifier")
    axes[1].legend(handles=[
        Line2D([0], [0], marker='o', color='w', label=cls,
               markerfacecolor=plt.cm.tab20(i / len(st.session_state.le_s_classes)), markersize=10)
        for i, cls in enumerate(st.session_state.le_s_classes)
    ], title="Classes")

    plt.tight_layout()
    st.pyplot(fig)
    
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
      
      


