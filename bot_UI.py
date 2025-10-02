import Main_classification
import classification_data
import streamlit as st
from Main_classification import render_sidebar
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import wikipedia
import re

# -------------------------
# Sidebar (persistent + context input)
# -------------------------
with st.sidebar:
    render_sidebar(
        Main_classification.training_text,
        Main_classification.training_pclass,
        Main_classification.training_sclass,
        Main_classification.accuracies
    )

    st.markdown("---")
    st.markdown("### Optional context passage")
    context_input = st.text_area("Paste your context here:", height=200)

context_input = context_input.strip() if context_input else ""

# -------------------------
# Chat panel
# -------------------------
st.markdown('<h1 style="font-size:70px">Welcome, User.</h1>', unsafe_allow_html=True)
st.markdown('<h1 style="font-size:30px">What\'s on today\'s agenda?</h1>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------
# Helper functions
# -------------------------
def clean_text(text):
    """Remove footnote markers like [a], [1], etc."""
    return re.sub(r'\[\w+\]', '', text)

def chunk_text(text, max_tokens=500):
    """Split text into chunks for the QA model (rough token count)."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    current_len = 0
    for s in sentences:
        s_len = len(s.split())
        if current_len + s_len > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = s + " "
            current_len = s_len
        else:
            current_chunk += s + " "
            current_len += s_len
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def fetch_wikipedia_context(query):
    """Fetch full Wikipedia page content."""
    try:
        page = wikipedia.page(query)
        return page.content.replace("\n", " ")
    except Exception:
        return ""

def answer_from_chunks(question, chunks, qa_pipeline):
    """Run QA on all chunks and pick the longest non-empty answer."""
    best_answer = ""
    for chunk in chunks:
        chunk_clean = clean_text(chunk)
        try:
            result = qa_pipeline(question=question, context=chunk_clean)
            answer = result.get("answer", "")
            if len(answer) > len(best_answer):
                best_answer = answer
        except Exception:
            continue
    return best_answer if best_answer else "No answer found."

# -------------------------
# Load extractive QA model from Hugging Face repo
# -------------------------
HF_REPO = "GryffinT/SQuAD.QA"  # your Hugging Face repo ID
tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
model = AutoModelForQuestionAnswering.from_pretrained(HF_REPO)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# -------------------------
# Chat input
# -------------------------
if prompt := st.chat_input("Ask Laurent anything."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. Classification
        classifications = Main_classification.pipeline.predict(prompt)

        # 2. Determine context
        if context_input:
            full_context = context_input
        else:
            full_context = fetch_wikipedia_context(prompt)
            if not full_context:
                full_context = "No context found for this question."

        # 3. Split into chunks
        chunks = chunk_text(full_context, max_tokens=500)

        # 4. Run QA on all chunks
        answer = answer_from_chunks(prompt, chunks, qa_pipeline)

        # 5. Display classifications + answer
        response = f"The classifications are: {classifications}, and my answer is {answer}"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
