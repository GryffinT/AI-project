import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModel
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import wikipedia as wiki
from wikipedia.exceptions import DisambiguationError
import spacy
import numpy as np
import torch

# -------------------------------
# Caching heavy model loads
# -------------------------------
@st.cache_resource
def load_models():
    CUSTOM_MODEL = "GryffinT/SQuAD.QA"
    
    tokenizer = AutoTokenizer.from_pretrained(CUSTOM_MODEL)

    # Try to load as QA model, else fallback
    is_qa_model = True
    try:
        model = AutoModelForQuestionAnswering.from_pretrained(CUSTOM_MODEL)
    except Exception as e:
        print("Warning: Could not load as QA model, loading as generic AutoModel. QA inference will be skipped.")
        print("Original error:", e)
        model = AutoModel.from_pretrained(CUSTOM_MODEL)
        is_qa_model = False

    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    nlp_model = spacy.load("en_core_web_sm")
    return tokenizer, model, encoder, nlp_model, is_qa_model

tokenizer, model, encoder, nlp, is_qa_model = load_models()

# -------------------------------
# Caching Wikipedia page lookups
# -------------------------------
@st.cache_data(ttl=3600)
def get_wiki_page(title):
    try:
        page = wiki.page(title)
        return page.content
    except DisambiguationError as e:
        try:
            page = wiki.page(e.options[0])
            return page.content
        except:
            return ""
    except:
        return ""

# -------------------------------
# Caching embeddings
# -------------------------------
@st.cache_data(ttl=3600)
def embed_text(text):
    return encoder.encode(text, convert_to_tensor=True).cpu().numpy()

# -------------------------------
# Main Output Function
# -------------------------------
def output(question: str, context: str) -> str:
    if context.strip():
        if is_qa_model:
            # Use QA model directly
            inputs = tokenizer(question, context, return_tensors="pt", truncation=True)
            outputs = model(**inputs)
            start_idx = torch.argmax(outputs.start_logits)
            end_idx = torch.argmax(outputs.end_logits)
            answer = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx + 1], skip_special_tokens=True)
            return answer
        else:
            # QA model not available
            return context[:600] + ("..." if len(context) > 600 else "")

    # Wikipedia fallback
    search_results = wiki.search(question)
    if not search_results:
        return "Apologies, it would seem there are no relevant sources for your inquiry."
    
    pages_data = []
    question_embed = torch.tensor(embed_text(question))
    q_doc = nlp(question)
    
    for page_title in search_results[:1]:  # top 1 result
        page_content = get_wiki_page(page_title)
        if not page_content:
            continue
        
        num_chunks = 3
        chunk_size = max(len(page_content) // num_chunks, 1)
        for i in range(num_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_chunks - 1 else len(page_content)
            chunk = page_content[start:end]
            if not chunk.strip():
                continue

            # TF-IDF similarity
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform([question, chunk])
            confidence_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

            # Semantic similarity
            chunk_embed = torch.tensor(embed_text(chunk))
            semantic_confidence = util.cos_sim(question_embed, chunk_embed).item()

            # Position score
            position_score = 1 - (i / (num_chunks - 1))

            # Entity overlap
            chunk_doc = nlp(chunk)
            question_entities = {ent.text for ent in q_doc.ents}
            chunk_entities = {ent.text for ent in chunk_doc.ents}
            normalized_overlap = len(question_entities & chunk_entities) / max(len(question_entities), 1)

            pages_data.append({
                "page_title": page_title,
                "chunk_text": chunk,
                "tfidf_score": confidence_score,
                "semantic_score": semantic_confidence,
                "position_score": position_score,
                "ent_score": normalized_overlap
            })

    if not pages_data:
        return "Apologies, it would seem there are no relevant sources for your inquiry."

    # Normalize scores
    def normalize_scores(pages_data, key):
        scores = np.array([p[key] for p in pages_data])
        if len(scores) == 0:
            return []
        min_val, max_val = scores.min(), scores.max()
        if max_val - min_val == 0:
            return [0.5] * len(scores)
        return (scores - min_val) / (max_val - min_val)
    
    for key in ["tfidf_score", "semantic_score", "position_score", "ent_score"]:
        normalized = normalize_scores(pages_data, key)
        for idx, val in enumerate(normalized):
            pages_data[idx][f"{key}_norm"] = val

    # Weighted combined score
    for p in pages_data:
        combined_score = (
            0.5 * p["semantic_score_norm"] +
            0.3 * p["tfidf_score_norm"] +
            0.1 * p["ent_score_norm"] +
            0.1 * p["position_score_norm"]
        )
        p["combined_score"] = combined_score

    # Softmax
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    combined_scores = np.array([p["combined_score"] for p in pages_data])
    softmax_scores = softmax(combined_scores)
    for idx, s in enumerate(softmax_scores):
        pages_data[idx]["final_confidence"] = s

    # Best chunk
    best_chunk = max(pages_data, key=lambda x: x["final_confidence"])
    best_chunk_text = best_chunk['chunk_text']

    # QA model inference (if available)
    if is_qa_model:
        inputs = tokenizer(question, best_chunk_text, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits)
        answer = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx + 1], skip_special_tokens=True)
        return f"{best_chunk['page_title']} — Confidence: {best_chunk['final_confidence']:.3f}\n\nAnswer: {answer}\n\nContext snippet: {best_chunk_text[:600]}"
    else:
        # QA model unavailable, just return snippet
        return f"{best_chunk['page_title']} — Confidence: {best_chunk['final_confidence']:.3f}\n\nContext snippet: {best_chunk_text[:600]}"
