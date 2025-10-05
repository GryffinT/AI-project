from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import wikipedia as wiki
import spacy
import numpy as np
import torch

# -------------------------------
# Load model from Hugging Face Hub
# -------------------------------
HF_REPO_ID = "GryffinT/SQuAD.QA"  # replace with your repo ID

tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
model = AutoModelForQuestionAnswering.from_pretrained(HF_REPO_ID)
encoder = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")

# -------------------------------
# Function to answer questions
# -------------------------------
def output(question: str, context: str) -> str:
    if not context.strip():
        search_results = wiki.search(question)
        if not search_results:
            return "Apologies, it would seem there are no relevant sources for your inquiry."
        else:
            pages_data = []
            question_embed = encoder.encode(question, convert_to_tensor=True)
            q_doc = nlp(question)

            for x in range(min(5, len(search_results))):
                page_title = search_results[x]
                wikiPage = wiki.page(page_title)
                page_content = wikiPage.content

                for i in range(3):
                    # Chunk setup
                    page_length = len(page_content)
                    chunk_size = page_length // 3
                    chunk_start = i * chunk_size
                    chunk_end = (i + 1) * chunk_size if i < 2 else page_length
                    chunk = page_content[chunk_start:chunk_end]

                    # TFIDF score
                    vectorizer = TfidfVectorizer(stop_words="english")
                    tfidf_matrix = vectorizer.fit_transform([question, chunk])
                    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
                    confidence_score = similarity[0][0]

                    # Semantic Score
                    chunk_embed = encoder.encode(chunk, convert_to_tensor=True)
                    semantic_similarity = util.cosine_similarity(question_embed, chunk_embed)
                    semantic_confidence = semantic_similarity.item()

                    # Position Score
                    position_score = 1 - (i / 2)

                    # Entity overlap
                    chunk_doc = nlp(chunk)
                    question_entities = {ent.text for ent in q_doc.ents}
                    chunk_entities = {ent.text for ent in chunk_doc.ents}
                    normalized_overlap = len(question_entities & chunk_entities) / max(len(question_entities), 1)

                    # Store data
                    pages_data.append({
                        "page": wikiPage,
                        "chunk_text": chunk,
                        "tfidf_score": confidence_score,
                        "semantic_score": semantic_confidence,
                        "position_score": position_score,
                        "ent_score": normalized_overlap
                    })

            # Normalization
            def normalize_scores(pages_data, key):
                scores = np.array([p[key] for p in pages_data])
                min_val, max_val = scores.min(), scores.max()
                if max_val - min_val == 0:
                    return [0.5] * len(scores)
                return (scores - min_val) / (max_val - min_val)

            for key in ["tfidf_score", "semantic_score", "position_score", "ent_score"]:
                normalized = normalize_scores(pages_data, key)
                for i, val in enumerate(normalized):
                    pages_data[i][f"{key}_norm"] = val

            # Combine metrics (weighted)
            for p in pages_data:
                combined_score = (
                    0.5 * p["semantic_score_norm"] +
                    0.3 * p["tfidf_score_norm"] +
                    0.1 * p["ent_score_norm"] +
                    0.1 * p["position_score_norm"]
                )
                p["combined_score"] = combined_score

            # Softmax normalization
            def softmax(x):
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum()

            combined_scores = np.array([p["combined_score"] for p in pages_data])
            softmax_scores = softmax(combined_scores)

            for i, s in enumerate(softmax_scores):
                pages_data[i]["final_confidence"] = s

            # Pick best chunk
            best_chunk = max(pages_data, key=lambda x: x["final_confidence"])
            return f"{best_chunk['page'].title} — Confidence: {best_chunk['final_confidence']:.3f}\n\n{best_chunk['chunk_text'][:600]}"

    else:
        # Use QA model when context is provided directly
        inputs = tokenizer(question, context, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits)
        answer = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx + 1])
        return answer

    
