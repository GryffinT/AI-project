import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
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
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    nlp_model = spacy.load("en_core_web_sm")
    return tokenizer, model, encoder, nlp_model

tokenizer, model, encoder, nlp = load_models()

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
    return encoder.encode(text, convert_to_tensor=True)

# -------------------------------
# Main Output Function
# -------------------------------
def output(question: str, context: str) -> str:
    if context.strip():
        # -------------------------------
        # Use QA model directly
        # -------------------------------
        inputs = tokenizer(question, context, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits)
        answer = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx + 1], skip_special_tokens=True)
        return answer
    else:
        # -------------------------------
        # Wikipedia-based search
        # -------------------------------
        search_results = wiki.search(question)
        if not search_results:
            return "Apologies, it would seem there are no relevant sources for your inquiry."

        pages_data = []
        question_embed = embed_text(question)
        q_doc = nlp(question)

        best_chunk = None
        max_attempts = 10  # safety limit to prevent infinite loops
        attempt = 0
        vectorizer = TfidfVectorizer(stop_words="english")
        
        while attempt < max_attempts:
            attempt += 1
            for page_title in search_results[:5]:
                title_embed = embed_text(page_title)
                title_semantic_score = util.cos_sim(question_embed, title_embed).item()
                print(f"The article {page_title} has a semantic score of: {title_semantic_score}")
                page_content = get_wiki_page(page_title)
                if not page_content.strip():
                    continue

                # Split into chunks
                num_chunks = 3
                chunk_size = max(len(page_content) // num_chunks, 1)
                for i in range(num_chunks):
                    start = i * chunk_size
                    end = (i + 1) * chunk_size if i < num_chunks - 1 else len(page_content)
                    chunk = page_content[start:end].strip()
                    if not chunk:
                        continue

                    # TF-IDF similarity
                    tfidf_matrix = vectorizer.fit_transform([question, chunk])
                    tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

                    # Semantic similarity
                    chunk_embed = embed_text(chunk)
                    semantic_score = util.cos_sim(question_embed, chunk_embed).item()

                    # Position score
                    position_score = 1 - (i / (num_chunks - 1))

                    # Entity overlap
                    chunk_doc = nlp(chunk)
                    question_entities = {ent.text for ent in q_doc.ents}
                    chunk_entities = {ent.text for ent in chunk_doc.ents}
                    ent_score = len(question_entities & chunk_entities) / max(len(question_entities), 1)
                    print(f" Processing attempt {attempt + 1} Chunk {i + 1}")

                    pages_data.append({
                        "page_title": page_title,
                        "chunk_text": chunk,
                        "tfidf_score": tfidf_score,
                        "semantic_score": semantic_score,
                        "position_score": position_score,
                        "ent_score": ent_score,
                        "title_score": title_semantic_score
                    })
                    

            if not pages_data:
                return "Apologies, it would seem there are no relevant sources for your inquiry."

            # -------------------------------
            # Normalize chunk-based scores
            # -------------------------------
            def normalize_scores(pages_data, key):
                scores = np.array([p[key] for p in pages_data])
                if len(scores) == 0:
                    return [0.5] * len(pages_data)
                min_val, max_val = scores.min(), scores.max()
                if max_val - min_val == 0:
                    return [0.5] * len(scores)
                return (scores - min_val) / (max_val - min_val)
            
            # Normalize chunk-level scores
            for key in ["semantic_score", "tfidf_score", "position_score", "ent_score"]:
                normalized = normalize_scores(pages_data, key)
                for idx, val in enumerate(normalized):
                    pages_data[idx][f"{key}_norm"] = val
            
            print("\nNormalized chunk-level scores (first 5 chunks):")
            for p in pages_data[:5]:
                print(f"Chunk from '{p['page_title'][:30]}...': semantic={p['semantic_score_norm']:.3f}, tfidf={p['tfidf_score_norm']:.3f}, ent={p['ent_score_norm']:.3f}, position={p['position_score_norm']:.3f}")
            
            # -------------------------------
            # Normalize title scores across pages
            # -------------------------------
            page_title_scores = {}
            for p in pages_data:
                # store the max title_score per page
                page_title_scores[p["page_title"]] = max(page_title_scores.get(p["page_title"], 0), p["title_score"])
            
            min_title, max_title = min(page_title_scores.values()), max(page_title_scores.values())
            
            for p in pages_data:
                if max_title - min_title == 0:
                    p["title_score_norm"] = 1.0
                else:
                    p["title_score_norm"] = (p["title_score"] - min_title) / (max_title - min_title)
            
            print("\nNormalized title scores per page:")
            for title, score in page_title_scores.items():
                normalized = (score - min_title) / (max_title - min_title) if max_title - min_title != 0 else 1.0
                print(f"'{title[:40]}...': {normalized:.3f}")
            
            # -------------------------------
            # Compute combined score
            # -------------------------------
            for p in pages_data:
                combined_score = (
                    0.35 * p["semantic_score_norm"] +
                    0.15 * p["tfidf_score_norm"] +
                    0.15 * p["ent_score_norm"] +
                    0.05 * p["position_score_norm"] +
                    0.30 * p["title_score_norm"]
                )
                p["combined_score"] = combined_score

            # Softmax for probabilistic confidence
            combined_scores = np.array([p["combined_score"] for p in pages_data])
            e_x = np.exp(combined_scores - np.max(combined_scores))
            softmax_scores = e_x / e_x.sum()
            print(f"The Softmax scores are: {softmax_scores}")
            for idx, s in enumerate(softmax_scores):
                pages_data[idx]["final_confidence"] = s

            # Select best chunk
            best_chunk = max(pages_data, key=lambda x: x["final_confidence"])

            # Stop looping if softmax confidence ≥ 0.5
            if best_chunk["final_confidence"] >= 0.5:
                break

        # Return the best chunk
        return f"{best_chunk['page_title']} — Confidence: {best_chunk['final_confidence']:.3f}\n\n{best_chunk['chunk_text'][:600]}"

