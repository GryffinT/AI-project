
# Neccisary dependencies
import os
import streamlit as st
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression # For the secondary and primary classifications.
from sklearn.model_selection import train_test_split # For creating the training and testing variables/shuffling data.
from sklearn.metrics import accuracy_score # For scoring the model's prediction accuracy at the end.
from sentence_transformers import SentenceTransformer # Importing the LLM, can be seen on line 22.
import numpy as np
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from classification_data import data
# Snatched these imports from HuggingFace directly, cause I cant ping them for some reason.
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# LLM model initiation through SentenceTransformers
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') I tried using this but I cant ping them so I have to do it manually, the following code is taken from their page for this model.
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Tokenize sentences
encoded_input = tokenizer(list(data.keys()), padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling
embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
embeddings = embeddings.cpu().numpy()  # Convert torch tensor to numpy array



# Filling texts variable by compiling a list of the keys from the data dict, such that the list is composed of the sentences.

#texts = list(data.keys()) # done above

# Actually looks like: ["What is Python?", "Who was Ghengis Khan?", "How can I study more effectively?", ...]

# Creating the embeddings, this uses the LLM from above through SentenceTransoformers to create dense vector arrays rather than sparse ones (TF-IDF) such that all values are non-zero floats, they're also NumPy arrays!
# This could potentially be an issue in that the dense vector arrays have higher memory usage than sparse.
# Also the encoded vectors are 2D, such that similar components are close in vector space for semantically similar sentences (clustered), on a 384-dimensional vector (READ).

#embeddings = model.encode(texts)  done above

# This turns the data texts into this:
# embeddings =
#array([
# [ 0.12, -0.34,  0.88, ...,  0.05],  # "What is Python?"
# [ 0.09, -0.30,  0.80, ...,  0.02],  # "Who was Ghengis Khan?"
# [ 0.50,  0.12, -0.20, ...,  0.11],  # "How can I study more effectively?"
# [ 0.48,  0.10, -0.22, ...,  0.08],  # "What time should I go to bed?"
# [ 0.15, -0.20,  0.95, ...,  0.12],  # "Whats the factored form of 2x^2 + 3x - 5"
# [ 0.18, -0.18,  0.92, ...,  0.09]   # "How can I solve for x in 7x + 8 = 5"
#])

# You can see that sentences with similar classifications are given ~similar values across sections (vertically).
# Note thanks Chat-GPT for encoding the embeddings for me for this example!

# Pretty standard stuff here, list comprehension that gets all of the values of the data dict at keys "pclass", effectivley aggregating a pclass list.

# ======= Prepare label arrays =======
primary_labels = [doc["pclass"] for doc in data.values()]
secondary_labels = [doc["sclass"] for doc in data.values()]

profane_labels = [1 if doc["Profane"]=="Yes" else 0 for doc in data.values()]
writing_labels = [1 if doc["Writing"]=="Yes" else 0 for doc in data.values()]
context_labels = [1 if doc["Context"]=="Yes" else 0 for doc in data.values()]

# ======= Train/test split =======
# For binary labels, stratify using one of them to ensure class balance
training_text, testing_text, training_profanity, testing_profanity, \
training_writing, testing_writing, training_context, testing_context, \
training_pclass, testing_pclass, training_sclass, testing_sclass = train_test_split(
    embeddings,
    profane_labels,
    writing_labels,
    context_labels,
    primary_labels,
    secondary_labels,
    test_size=0.1,
    random_state=42,
    stratify=profane_labels  # ensures at least some '1' and '0' in training set
)

# ======= Setup label dictionary =======
label_sets = {
    "primary": (training_pclass, testing_pclass),
    "secondary": (training_sclass, testing_sclass),
    "profanity": (training_profanity, testing_profanity),
    "writing": (training_writing, testing_writing),
    "context": (training_context, testing_context)
}

classifiers = {}
accuracies = {}

# ======= Train classifiers =======
for name, (y_train, y_test) in label_sets.items():
    # Use DummyClassifier if only one class exists
    if len(np.unique(y_train)) > 1:
        clf = LogisticRegression(max_iter=500)
    else:
        clf = DummyClassifier(strategy="constant", constant=y_train[0])
    
    clf.fit(training_text, y_train)
    pred = clf.predict(testing_text)
    
    classifiers[name] = clf
    accuracies[name] = accuracy_score(y_test, pred) * 100

# ======= Print results =======
for label, acc in accuracies.items():
    print(f"{label.capitalize()} Accuracy: {acc:.2f}%")

st.title("AI Project")

# -------------------------
# Ensure embeddings and labels are aligned
n = min(len(training_text), len(training_pclass), len(training_sclass))
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
for label, acc in accuracies.items():
    st.write(f"The model's {label} accuracy is operating at {acc:.2f}%")

# ======= TextClassifier =======
class TextClassifier:
    def __init__(self, tokenizer, model, classifiers_dict):
        """
        classifiers_dict: dictionary containing trained classifiers for each label
        e.g., classifiers_dict = {
            'primary': clf_primary,
            'secondary': clf_secondary,
            'profanity': clf_profanity,
            'writing': clf_writing,
            'context': clf_context
        }
        """
        self.tokenizer = tokenizer
        self.model = model
        self.classifiers = classifiers_dict

    def embed(self, texts):
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def predict(self, texts):
        emb = self.embed([texts])  # Wrap text in a list to handle single input
        preds = {}
        for name, clf in self.classifiers.items():
            pred = clf.predict(emb)[0]
            preds[name] = str(pred)
        # Return a formatted string
        return ", ".join([f"{k}: {v}" for k, v in preds.items()])

# ======= Usage in Streamlit =======
pipeline = TextClassifier(tokenizer, model, classifiers)

statement = st.text_input("Enter a statement to the system for classification")
if st.button("Classify"):
    classifications = pipeline.predict(statement)
    st.markdown(f"The classifications are: {classifications}.")
