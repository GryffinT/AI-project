# app.py
import re
import streamlit as st
import wikipedia
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import Main_classification
import classification_data
from Main_classification import render_sidebar

# -------------------------
# Config
# -------------------------
HF_REPO = "GryffinT/SQuAD.QA"  # <-- your Hugging Face repo id
CHUNK_MAX_WORDS = 700         # size of each chunk in words
CHUNK_OVERLAP = 120           # overlap between chunks in words
SCORE_THRESHOLD = 0.0         # include answers with score >= this
MAX_AGG_ANSWERS = 5          # max number of distinct answers to include
MAX_FINAL_CHARS = 1000       # truncate final aggregated answer to this many chars

# -------------------------
# Sidebar: render sidebar + context input
# -------------------------
with st.sidebar:
    render_sidebar(
        Main_classification.training_text,
        Main_classification.training_pclass,
        Main_classification.training_sclass,
        Main_classification.accuracies
    )
    st.markdown("---")
    st.markdown("### Optional context passage (sidebar)")
    context_input = st.text_area("Paste your context here (optional):", height=220)
    st.markdown("---")
    debug_mode = st.checkbox("Show debug info (context, chunks)", value=False)

context_input = context_input.strip() if context_input else ""

# -------------------------
# Utilities
# -------------------------
def clean_text(text: str) -> str:
    """
    Clean wikipedia text: remove bracketed notes like [1], [a], etc., and normalize whitespace.
    """
    if not text:
        return ""
    text = re.sub(r'\[([^\]]+)\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fetch_wikipedia_content(query: str) -> str:
    """
    Fetch the full wikipedia page content for the query.
    Handles disambiguation by picking the first sensible result.
    """
    try:
        search_results = wikipedia.search(query, results=5)
        if not search_results:
            return ""
        title = search_results[0]
        try:
            page = wikipedia.page(title, auto_suggest=False)
            return page.content
        except wikipedia.DisambiguationError as e:
            choice = e.options[0] if e.options else title
            try:
                page = wikipedia.page(choice, auto_suggest=False)
                return page.content
            except Exception:
                return ""
        except wikipedia.PageError:
            return ""
    except Exception:
        return ""

def chunk_text_overlap(text: str, max_words: int = CHUNK_MAX_WORDS, overlap: int = CHUNK_OVERLAP):
    """
    Split text into chunks of roughly `max_words` words with `overlap` words between chunks.
    Returns list of chunk strings.
    """
    if not text:
        return []
    words = text.split()
    chunks = []
    start = 0
    n = len(words)
    while start < n:
        end = min(start + max_words, n)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == n:
            break
        start += max_words - overlap
    return chunks

def aggregate_answers_with_subject(question: str, chunks: list, qa_pipeline, 
                                   score_threshold: float = SCORE_THRESHOLD, max_answers: int = MAX_AGG_ANSWERS):
    """
    Run QA pipeline over chunks and aggregate answers, ensuring the question subject appears in the output.
    """
    candidates = []
    question_keywords = set(q.lower() for q in re.findall(r'\w+', question))
    
    for chunk in chunks:
        chunk_clean = clean_text(chunk)
        if not chunk_clean:
            continue
        try:
            res = qa_pipeline(question=question, context=chunk_clean)
        except Exception:
            continue
        ans = res.get("answer", "").strip()
        score = float(res.get("score", 0.0))
        if ans and score >= score_threshold:
            ans_lower = ans.lower()
            if any(kw in ans_lower for kw in question_keywords):
                candidates.append((ans, score))
            else:
                # downweight unrelated answer
                candidates.append((ans, score * 0.5))
    
    if not candidates:
        return "No answer found."
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    unique = []
    seen = set()
    for ans, score in candidates:
        if ans in seen:
            continue
        seen.add(ans)
        unique.append((ans, score))
        if len(unique) >= max_answers:
            break
    
    aggregated = " ".join([a for a, s in unique])
    if len(aggregated) > MAX_FINAL_CHARS:
        aggregated = aggregated[:MAX_FINAL_CHARS].rsplit(" ", 1)[0] + "..."
    return aggregated

# -------------------------
# Load model & pipeline
# -------------------------
@st.cache_resource(show_spinner=False)
def load_qa_pipeline(repo_id: str):
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForQuestionAnswering.from_pretrained(repo_id)
    qa = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa

try:
    qa_pipeline = load_qa_pipeline(HF_REPO)
except Exception as e:
    st.error(f"Failed to load model from Hugging Face repo '{HF_REPO}': {e}")
    st.stop()

# -------------------------
# UI header & chat state
# -------------------------
st.markdown('<h1 style="font-size:70px">Welcome, User.</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="font-size:30px">Ask Laurent anything.</h2>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------
# Chat input handling
# -------------------------
if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1) classification
        try:
            classifications = Main_classification.pipeline.predict(prompt)
        except Exception as e:
            classifications = f"Classification failed: {e}"

        # 2) determine context: user-provided or wikipedia full page
        if context_input:
            full_context = context_input
        else:
            raw = fetch_wikipedia_content(prompt)
            full_context = raw if raw else ""

        if not full_context:
            answer = "No context available (no user context and Wikipedia fetch failed)."
        else:
            cleaned = clean_text(full_context)
            if debug_mode:
                st.write("---- DEBUG: cleaned context preview ----")
                st.write(cleaned[:2000])
                st.write("---- end preview ----")
            chunks = chunk_text_overlap(cleaned, max_words=CHUNK_MAX_WORDS, overlap=CHUNK_OVERLAP)
            if debug_mode:
                st.write(f"Chunks created: {len(chunks)}")
            answer = aggregate_answers_with_subject(prompt, chunks, qa_pipeline)

        # 3) final response assembly
        response = f"The classifications are: {classifications}, and my answer is {answer}"
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
