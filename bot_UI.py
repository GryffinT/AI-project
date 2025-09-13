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

with st.sidebar:
        # ======= Print results =======
    for label, acc in accuracies.items(): # iterates and gathers the accuracies and their respective predictors from before
        print(f"{label.capitalize()} Accuracy: {acc:.2f}%") # prints out the accuracies of each predictor
    
    # -------------------------
    # Ensure embeddings and labels are aligned
    
    # Confession time, I'm not so well versed in this graphing stuff and... kinda... didint write it... its only temporary for diagnostic purposes so... yeah.
    
    n = min(len(training_text), len(training_pclass), len(training_sclass)) #
    embeddings_plot = np.array(training_text[:n])
    p_labels = np.array(training_pclass[:n])
    s_labels = np.array(training_sclass[:n])
    
    # -------------------------
    # Encode labels for coloring
    le_p = LabelEncoder()
    p_labels_encoded = le_p.fit_transform(p_labels)
    
    le_s = LabelEncoder()
    s_labels_encoded = le_s.fit_transform(s_labels)
    
    # -------------------------
    # Reduce embeddings to 2D
    X_pca = PCA(n_components=2).fit_transform(embeddings_plot)
    
    # -------------------------
    # Create side-by-side scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    
    # Primary classifier scatter
    scatter1 = axes[0].scatter(X_pca[:,0], X_pca[:,1], c=p_labels_encoded, cmap='tab10',
                               alpha=0.7, edgecolor='k')
    axes[0].set_title("Primary Classifier")
    axes[0].set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")
    
    # Legend for primary
    colors = plt.cm.tab10(np.linspace(0, 1, len(le_p.classes_)))
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=cls,
                              markerfacecolor=colors[i], markersize=10)
                       for i, cls in enumerate(le_p.classes_)]
    axes[0].legend(handles=legend_elements, title="Classes")
    
    # Secondary classifier scatter
    scatter2 = axes[1].scatter(X_pca[:,0], X_pca[:,1], c=s_labels_encoded, cmap='tab20',
                               alpha=0.7, edgecolor='k')
    axes[1].set_title("Secondary Classifier")
    axes[1].set_xlabel("PC 1")
    axes[1].set_ylabel("PC 2")
    
    # Legend for secondary
    colors2 = plt.cm.tab20(np.linspace(0, 1, len(le_s.classes_)))
    legend_elements2 = [Line2D([0], [0], marker='o', color='w', label=cls,
                               markerfacecolor=colors2[i], markersize=10)
                        for i, cls in enumerate(le_s.classes_)]
    axes[1].legend(handles=legend_elements2, title="Classes")
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # ======= Display accuracies dynamically =======
    for label, acc in accuracies.items(): # gets the accuracy rates per predictor just like before
        st.write(f"The model's {label} accuracy is operating at {acc:.2f}%") # displays the accuracy of each predictor as a writeup on streamlit's interface.
if prompt := st.chat_input("Ask me anything"):
  with st.chat_message("user"):
    st.markdown(prompt)
  st.session_state.messages.append({"role": "user", "content": prompt})

