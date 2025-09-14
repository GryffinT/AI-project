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

# End of embeds

# ======= Prepare label arrays =======

# Multi-class labels (LogReg)

primary_labels = [doc["pclass"] for doc in data.values()] # List comprehension to initialize the primary_labels list and then populate it with all of the values associated with the pclass label from the data.
secondary_labels = [doc["sclass"] for doc in data.values()] # Same as the primary labels list, just sclass this time.

# Binary labels

profane_labels = [1 if doc["Profane"]=="Yes" else 0 for doc in data.values()] # List comprehension, initializes profane_labels list and populates each index with either 1, if the value of "Profane" is "Yes" or 0 if "No"
writing_labels = [1 if doc["Writing"]=="Yes" else 0 for doc in data.values()] # Same as previous but for "Writing".
context_labels = [1 if doc["Context"]=="Yes" else 0 for doc in data.values()] # ^ ("Context")

# ======= Train/test split =======

# Test/Train/Split, shuffles the data give, so the embedded text stored in "training_text", multi-class labels from above and the dummy labels as well.
stratify_labels = np.array([f"{p}-{w}-{c}" for p, w, c in zip(profane_labels, writing_labels, context_labels)])
# For binary labels, stratify using one of them to ensure class balance
training_text, testing_text, training_profanity, testing_profanity, \
training_writing, testing_writing, training_context, testing_context, \
training_pclass, testing_pclass, training_sclass, testing_sclass = train_test_split(
    embeddings, # Embedded text from the embed portion.
    profane_labels, # line 56
    writing_labels, # line 57
    context_labels, # line 58
    primary_labels, # line 51
    secondary_labels, # line 52
    test_size=0.25, # 25% of the shuffled data is reserved for the training datasets to gauge model success.
    random_state=42, # randomized shuffle seed.
    stratify=profane_labels,  # ensures at least some '1' and '0' in training set
)

# ======= Setup label dictionary =======

# This is kind of just clean up so that I can ref the labels AND have the ability to easily add new features.

label_sets = { 
    "primary": (training_pclass, testing_pclass), # Primary labels are stored in the training_pclass set and testing p_class set
    "secondary": (training_sclass, testing_sclass), # ^ (sclass)
    "profanity": (training_profanity, testing_profanity), # ^ (profanity)
    "writing": (training_writing, testing_writing), # ^ (writing)
    "context": (training_context, testing_context) # ^ (context)
} 

classifiers = {} # initialize empy classifiers dict
accuracies = {} # ^ accuracies

# ======= Train classifiers =======
for name, (y_train, y_test) in label_sets.items(): #  Iterates over the label_sets dict and gets the labels for a given label set (Primary, Secondary, etc)
    clf = LogisticRegression(max_iter=500) # Run the LogReg with a maximum iteration of 500 times, which it will stop short if the loss converges early.

    clf.fit(training_text, y_train) # Classifier, learn from (fit) the training_text and the training labels.
    pred = clf.predict(testing_text) # Now, classifier, predict from this testing text (10% atm)
    
    classifiers[name] = clf # This classifier (Primary, secondary, whatever) is saved to the list classifers under the name of name (secondary, profanity, etc)
    accuracies[name] = accuracy_score(y_test, pred) * 100 # Just a calculation to find accuracy, compares the testing labels to the predicted labels. Then saves the accuracies to the accuracies list under the pred's name

# ======= Print results =======
for label, acc in accuracies.items(): # iterates and gathers the accuracies and their respective predictors from before
    print(f"{label.capitalize()} Accuracy: {acc:.2f}%") # prints out the accuracies of each predictor



# ======= Render side bar =======

def render_sidebar(training_text, training_pclass, training_sclass, accuracies):
    # Compute/store data only once
    if "data_persistent" not in st.session_state:
        n = min(len(training_text), len(training_pclass), len(training_sclass))
        st.session_state.embeddings_plot = np.array(training_text[:n])
        st.session_state.p_labels = np.array(training_pclass[:n])
        st.session_state.s_labels = np.array(training_sclass[:n])

        # Encode labels
        le_p = LabelEncoder()
        st.session_state.p_labels_encoded = le_p.fit_transform(st.session_state.p_labels)
        st.session_state.p_classes = le_p.classes_

        le_s = LabelEncoder()
        st.session_state.s_labels_encoded = le_s.fit_transform(st.session_state.s_labels)
        st.session_state.s_classes = le_s.classes_

        # PCA
        st.session_state.X_pca = PCA(n_components=2).fit_transform(st.session_state.embeddings_plot)

        # Store accuracies
        st.session_state.accuracies = accuracies

        st.session_state.primary_labels = primary_labels
        st.session_state.secondary_labels = secondary_labels
        st.session_state.profane_labels = profane_labels 
        st.session_state.writing_labels = writing_labels
        st.session_state.context_labels = context_labels


        # Mark persistent data ready
        st.session_state.data_persistent = True

    # Render sidebar using persistent data
    with st.sidebar:
        st.markdown('<h1 style="font-size:30px">LauRenT.16</h1>', unsafe_allow_html=True)
        st.markdown('<h1 style="font-size:15px">A Logistic Regression Transformer with Float Point 16 percision.</h1>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].scatter(st.session_state.X_pca[:, 0], st.session_state.X_pca[:, 1],
                        c=st.session_state.p_labels_encoded, cmap='tab10', alpha=0.7, edgecolor='k')
        axes[0].set_title("Primary Classifier")
        axes[0].set_xlabel("PC 1")
        axes[0].set_ylabel("PC 2")
        colors = plt.cm.tab10(np.linspace(0, 1, len(st.session_state.p_classes)))
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=cls,
                                  markerfacecolor=colors[i], markersize=10)
                           for i, cls in enumerate(st.session_state.p_classes)]
        axes[0].legend(handles=legend_elements, title="Classes")

        axes[1].scatter(st.session_state.X_pca[:, 0], st.session_state.X_pca[:, 1],
                        c=st.session_state.s_labels_encoded, cmap='tab20', alpha=0.7, edgecolor='k')
        axes[1].set_title("Secondary Classifier")
        axes[1].set_xlabel("PC 1")
        axes[1].set_ylabel("PC 2")
        colors2 = plt.cm.tab20(np.linspace(0, 1, len(st.session_state.s_classes)))
        legend_elements2 = [Line2D([0], [0], marker='o', color='w', label=cls,
                                   markerfacecolor=colors2[i], markersize=10)
                            for i, cls in enumerate(st.session_state.s_classes)]
        axes[1].legend(handles=legend_elements2, title="Classes")

        plt.tight_layout()
        st.pyplot(fig)

        # Display persistent accuracies
        for label, acc in st.session_state.accuracies.items():
            st.progress((acc / 100), text=f"{label.capitalize()} Accuracy at {acc:.2f}%")
            
        profane_true = [doc for doc in st.session_state["profane_labels"] if doc == 1]
        context_true = [doc for doc in st.session_state["context_labels"] if doc == 1]
        writing_true = [doc for doc in st.session_state["writing_labels"] if doc == 1]

        st.metric("Primary Classifications", len(st.session_state["primary_labels"]), len(st.session_state["primary_labels"]))
        st.metric("Secondary Classifications", len(st.session_state["secondary_labels"]), len(st.session_state["secondary_labels"]))
        st.metric("Profane Classifications", len(profane_true), len(profane_true))
        st.metric("Writing Classifications", len(writing_true), len(writing_true))
        st.metric("Context Classifications", len(context_true), len(context_true))

# ======= TextClassifier =======
class TextClassifier: # OOP python... scary. This makes the TextClassifier class
    def __init__(self, tokenizer, model, classifiers_dict): # Initializes the object
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
        self.tokenizer = tokenizer # Self reference to the tokenizer, basically linking it to the tokenizer from before
        self.model = model # ^
        self.classifiers = classifiers_dict # ^ 

    def embed(self, texts): # re-worked embed function within the class to handle user input. 
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt') # encodes the input text with padding, truncation, and as a PyTorch Tensor (fancy vector array)
        with torch.no_grad(): # Within the torch context manager dont track gradients (saves memory)
            model_output = self.model(**encoded_input) # the output, passes the encoded tensor from before into the LogReg model, also ** means argument unpacking so it shows the content, kinda like a file unzip.
        embeddings = mean_pooling(model_output, encoded_input['attention_mask']) # Pretty much just averages the attention masked encoded inputs and the model outputs right above it, more on this below.
        # The attention_masked means that during the encoding process, because we used a transformer the semantic meaning of the words were only RETROSPECIVE. so meaning was correlated retrospectivley.
        # The model_output is the result of passing the input through the LogReg
        # the mean_pooling is actually just gathering the mean/average of these values to save computational power at the cost of accuracy
        embeddings = F.normalize(embeddings, p=2, dim=1) # L2 normalization of the embed tensors into magnitude 1 vectors.
        return embeddings.cpu().numpy() # return the embeddings in NumPy format on the CPU.

    def predict(self, texts): # Predict function
        emb = self.embed([texts])  # Wrap text in a list to handle single input 
        preds = {} # Empty prediction list.
        for name, clf in self.classifiers.items(): # Iterator that for each name and classifier in the classifiers it makes a prediction for the respective classifiers using the embed.
            pred = clf.predict(emb)[0] # ^
            preds[name] = str(pred) # appends the prediction as a string to the preds list under the respective name.
        # Return a formatted string
        return ", ".join([f"{k}: {v}" for k, v in preds.items()]) # returns the joined predictions from the preds list 

# ======= Usage in Streamlit =======
pipeline = TextClassifier(tokenizer, model, classifiers) # establishes the pipeline (not really though)
