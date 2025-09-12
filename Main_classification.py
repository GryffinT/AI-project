
            # Neccisary dependencies
import os
import streamlit as st
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression # For the secondary and primary classifications.
from sklearn.model_selection import train_test_split # For creating the training and testing variables/shuffling data.
from sklearn.metrics import accuracy_score # For scoring the model's prediction accuracy at the end.
from sentence_transformers import SentenceTransformer # Importing the LLM, can be seen on line 22.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# Snatched these imports from HuggingFace directly, cause I cant ping them for some reason.
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

            # Training and testing data, Dict of Dict, Text -> Primary classification -> Secondary classification
            # subsequent (tertiary, quartinary, etc) classifications can be added following the format and adjusting accordingly

data = {
    # Computer Science / Research
    "What is Python?": {"pclass": "Computer Science", "sclass": "Research/Informative"},
    "How does a binary search algorithm work?": {"pclass": "Computer Science", "sclass": "Research/Informative"},
    "What is the difference between Java and JavaScript?": {"pclass": "Computer Science", "sclass": "Research/Informative"},
    "Explain object-oriented programming concepts.": {"pclass": "Computer Science", "sclass": "Research/Informative"},
    "What are the key features of Linux?": {"pclass": "Computer Science", "sclass": "Research/Informative"},
    "How does a neural network learn?": {"pclass": "Computer Science", "sclass": "Research/Informative"},
    "What is the function of an operating system?": {"pclass": "Computer Science", "sclass": "Research/Informative"},
    "How do databases store information?": {"pclass": "Computer Science", "sclass": "Research/Informative"},
    "Explain recursion with an example.": {"pclass": "Computer Science", "sclass": "Research/Informative"},
    "What is a hash table and how does it work?": {"pclass": "Computer Science", "sclass": "Research/Informative"},
    "Describe the differences between TCP and UDP.": {"pclass": "Computer Science", "sclass": "Research/Informative"},
    "What are some common programming paradigms?": {"pclass": "Computer Science", "sclass": "Research/Informative"},
    "How do compilers convert code into machine language?": {"pclass": "Computer Science", "sclass": "Research/Informative"},
    "What is the purpose of version control systems?": {"pclass": "Computer Science", "sclass": "Research/Informative"},
    "Explain cloud computing in simple terms.": {"pclass": "Computer Science", "sclass": "Research/Informative"},

    # History / Research
    "Who was Ghengis Khan?": {"pclass": "History", "sclass": "Research/Informative"},
    "What caused the fall of the Roman Empire?": {"pclass": "History", "sclass": "Research/Informative"},
    "Explain the significance of the French Revolution.": {"pclass": "History", "sclass": "Research/Informative"},
    "Who was Cleopatra?": {"pclass": "History", "sclass": "Research/Informative"},
    "What were the main causes of World War I?": {"pclass": "History", "sclass": "Research/Informative"},
    "Describe the Renaissance period in Europe.": {"pclass": "History", "sclass": "Research/Informative"},
    "What led to the American Civil War?": {"pclass": "History", "sclass": "Research/Informative"},
    "Who was Napoleon Bonaparte?": {"pclass": "History", "sclass": "Research/Informative"},
    "Explain the Cold War in simple terms.": {"pclass": "History", "sclass": "Research/Informative"},
    "What was the Industrial Revolution?": {"pclass": "History", "sclass": "Research/Informative"},
    "Who signed the Magna Carta?": {"pclass": "History", "sclass": "Research/Informative"},
    "Describe the causes of the French and Indian War.": {"pclass": "History", "sclass": "Research/Informative"},
    "What was the significance of the Berlin Wall?": {"pclass": "History", "sclass": "Research/Informative"},
    "Who were the Vikings and where did they travel?": {"pclass": "History", "sclass": "Research/Informative"},
    "What is the history of the Silk Road?": {"pclass": "History", "sclass": "Research/Informative"},

    # Self Help / Advice
    "How can I study more effectively?": {"pclass": "Self Help", "sclass": "Advice/Guidance"},
    "What time should I go to bed?": {"pclass": "Self Help", "sclass": "Advice/Guidance"},
    "How do I improve my focus while working?": {"pclass": "Self Help", "sclass": "Advice/Guidance"},
    "How can I manage my stress better?": {"pclass": "Self Help", "sclass": "Advice/Guidance"},
    "What are good habits for productivity?": {"pclass": "Self Help", "sclass": "Advice/Guidance"},
    "How can I improve my memory?": {"pclass": "Self Help", "sclass": "Advice/Guidance"},
    "What are effective time management strategies?": {"pclass": "Self Help", "sclass": "Advice/Guidance"},
    "How do I stay motivated to exercise?": {"pclass": "Self Help", "sclass": "Advice/Guidance"},
    "What are tips for better concentration?": {"pclass": "Self Help", "sclass": "Advice/Guidance"},
    "How can I build confidence?": {"pclass": "Self Help", "sclass": "Advice/Guidance"},
    "How can I improve my public speaking skills?": {"pclass": "Self Help", "sclass": "Advice/Guidance"},
    "What are ways to maintain a positive mindset?": {"pclass": "Self Help", "sclass": "Advice/Guidance"},
    "How do I avoid procrastination?": {"pclass": "Self Help", "sclass": "Advice/Guidance"},
    "What are effective study techniques for exams?": {"pclass": "Self Help", "sclass": "Advice/Guidance"},
    "How can I better handle criticism?": {"pclass": "Self Help", "sclass": "Advice/Guidance"},

    # Mathematics / Teaching
    "Whats the factored form of 2x^2 + 3x - 5": {"pclass": "Mathematics", "sclass": "Teaching"},
    "How can I solve for x in 7x + 8 = 5": {"pclass": "Mathematics", "sclass": "Teaching"},
    "What is the derivative of x^2 + 3x?": {"pclass": "Mathematics", "sclass": "Teaching"},
    "How do you calculate the area of a circle?": {"pclass": "Mathematics", "sclass": "Teaching"},
    "Explain the Pythagorean theorem with an example.": {"pclass": "Mathematics", "sclass": "Teaching"},
    "How do you solve a quadratic equation?": {"pclass": "Mathematics", "sclass": "Teaching"},
    "What is an integral and how is it calculated?": {"pclass": "Mathematics", "sclass": "Teaching"},
    "Explain how to find the slope of a line.": {"pclass": "Mathematics", "sclass": "Teaching"},
    "What is the difference between permutation and combination?": {"pclass": "Mathematics", "sclass": "Teaching"},
    "How do you calculate the volume of a cylinder?": {"pclass": "Mathematics", "sclass": "Teaching"},
    "Explain the concept of limits in calculus.": {"pclass": "Mathematics", "sclass": "Teaching"},
    "How do you convert fractions to decimals?": {"pclass": "Mathematics", "sclass": "Teaching"},
    "What are prime numbers and how to identify them?": {"pclass": "Mathematics", "sclass": "Teaching"},
    "Explain the difference between mean, median, and mode.": {"pclass": "Mathematics", "sclass": "Teaching"},
    "How do you solve systems of linear equations?": {"pclass": "Mathematics", "sclass": "Teaching"}
}


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

primary_labels = [document["pclass"] for document in data.values()]

            # looks like: ["Computer Science", "History", "Self Help", ...]

            # Again, the same thing just for sclass this time.

secondary_labels = [document["sclass"] for document in data.values()]

            # Looks like: ["Research/Informative", "Research/Informative", "Advice/Guidance", ...]

            # Also for clarity, either could be re-written as:
            #
            # primary_labels = []
            # for document in data.values():
            #     primary_labels.append(document["pclass"])
            #
            # I used primary_labels as the example but you could also do secondary, here document is an arbitrary iterator variable and "pclass" is the key.
            #

            # Yay! More ML stuff, basically this is a split of the data from the data dict into training and testing portions.
            # You can see that it defines and fills the variables, "training_text", "testing_text", "training_pclass", "training_sclass", and "testing_sclass".
            # It fills those variables with a random shuffle (sort of random, the seed is 42, this can be seen in random_state=42) of the data from the data dict's labels as well as the embeddings (line 35)
            # Simply put, this takes test_size slices of the data parameters (embeddings, primary_labels, and secondary_labels), so 50/50 and shuffles it,
            #  assigning 50% of the data to training and 50% of the data to testing.

training_text, testing_text, training_pclass, testing_pclass, training_sclass, testing_sclass = train_test_split(embeddings, primary_labels, secondary_labels, test_size=.5, random_state=42)

            # I would show what it looks like but it'd be a massive pain..

            # Initiating the primary classifier, it uses LogisticRegression (sklearn.linear_mode, line 4) to map the input features to the class probabilities, being a fancy abstract-dimension and all (line 33).
            # The maximum # of optimization steps to take is 500 as per the max_iter=500 parameter, but it will stop short if the weights converge.
            # it's then fit to the training text and training primary classifications to... train off of.
            # Note, .fit loosely translates to "learn from" which I find easier, or more precisely "optimizes the weights of the logistic regression model to minimize classification error", thanks Chatty.
            # So basically its learning where to plot and cluster it's weights/classifications based off of the contents of the training_text and training_pclass (line 68). Simple right?

clf_primary = LogisticRegression(max_iter=500)
clf_primary.fit(training_text, training_pclass) # Hey, Primary Classifier, learn (recognize the patterns) from this training text EMBED I have and each vector's corresponding label from training_pclass.

            # Okay, so this really is just the same thing as the primary classifier, but it does it for the secondary classifications and uses sclass rather than pclass, for further detail refer to line 72.

clf_secondary = LogisticRegression(max_iter=500)
clf_secondary.fit(training_text, training_sclass) # Comment on line 79

            # Initiates the primary predictor, this calls on the primary classifier to predict through .predict using the testing text as a paramenter!
            # So, I dont completeley get it... but I think it calculates the linear scores along the LogReg line from the weights/biases and chooses the highest probability class.
                # Afternote: the input embedding is converted into a linear score for each class
                # (by taking the dot product with each class's learned weight vector and adding the bias). 
                # These scores are then converted into probabilities via softmax, 
                # and the class with the highest probability is chosen as the predicted label.
            # It then chooses the closest, and highest classification to the position found for the input text and chooses that classification.
            # basically, it finds where it WOULD plot the input data in following with its training data, and then chooses the highest classifications that have the closest distance from the weights as the chosen point.
            # This uses fancy math like softmax functions, which I know conceptually but not fundamentally, so I'm not terribly familiar.
            # If you want a better description here's GPT's take:

            # Uses the primary classifier to predict classes for the testing embeddings.
            # .predict() takes the testing text embeddings as input and outputs the most likely class labels.
            # Internally, the classifier computes a score for each class (using the learned weights and biases) and applies a softmax to estimate probabilities.
            # The class with the highest probability is chosen as the predicted label.
            # No retraining occurs here; it just applies what the classifier learned during .fit().


pred_primary = clf_primary.predict(testing_text) # Another mention real quick, testing_text is an embed, therefore its a dense vector array, NOT raw text, refer to lines 68, and 31.

            # Secondary predictor so refer to lines 107-114.

pred_secondary = clf_secondary.predict(testing_text) # Comment on line 114.

            # Standard procedure here, just prints the classification predictions as a percentage using the accuracy_score routine from sklearn.metrics on line 6.
            # Conceptually, it compares the output of the secondary/primary predictors agains the true classifications for the inputs used in the predictors. (var def: line 68)
primary_accuracy = accuracy_score(testing_pclass, pred_primary) * 100
secondary_accuracy = accuracy_score(testing_sclass, pred_secondary) * 100

st.title("AI Project")

# Ensure embeddings and labels are aligned
n = min(len(embeddings), len(training_pclass), len(training_sclass))
embeddings = embeddings[:n]
p_labels = np.array(training_pclass[:n])
s_labels = np.array(training_sclass[:n])

# Encode labels for plotting
le_p = LabelEncoder()
p_labels_encoded = le_p.fit_transform(p_labels)

le_s = LabelEncoder()
s_labels_encoded = le_s.fit_transform(s_labels)

# Reduce embeddings to 2D for plotting
X_pca = PCA(n_components=2).fit_transform(embeddings)

# -------------------------
# Create side-by-side figure
fig, axes = plt.subplots(1, 2, figsize=(14,6))

# Primary classifier scatter
scatter1 = axes[0].scatter(X_pca[:,0], X_pca[:,1], c=p_labels_encoded, cmap='tab10', alpha=0.7, edgecolor='k')
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
scatter2 = axes[1].scatter(X_pca[:,0], X_pca[:,1], c=s_labels_encoded, cmap='tab20', alpha=0.7, edgecolor='k')
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

st.write(f"The model's primary accuracy is operating at {primary_accuracy}%")
st.write(f"The model's secondary accuracy is operating at {secondary_accuracy}%")

class TextClassifier:
    def __init__(self, tokenizer, model, clf_primary, clf_secondary):
        self.tokenizer = tokenizer
        self.model = model
        self.clf_primary = clf_primary
        self.clf_secondary = clf_secondary

    def embed(self, texts):
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def predict(self, texts):
                emb = self.embed(texts)
                p_pred = self.clf_primary.predict(emb)[0]
                s_pred = self.clf_secondary.predict(emb)[0]
                # Get predictions from your pipeline
                preds = [p_pred, s_pred]  # e.g., ['History', 'Research/Informative']
            
                # Ensure it's a list of Python strings
                preds_clean = [str(p) for p in preds]
            
                # Join into a nice string for display
                return (", ".join(preds_clean))


# Usage:
pipeline = TextClassifier(tokenizer, model, clf_primary, clf_secondary)
statement = st.text_input("Enter a statment to the system for classification")
if st.button("Classify"):
            st.markdown(pipeline.predict(statement))
