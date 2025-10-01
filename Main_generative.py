from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# -------------------------------
# Load model from Hugging Face Hub
# -------------------------------
HF_REPO_ID = "GryffinT/SQuAD.QA"  # Replace with your HF repo ID

tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
model = AutoModelForQuestionAnswering.from_pretrained(HF_REPO_ID)

# -------------------------------
# Function to answer questions
# -------------------------------
def answer_question(question: str, context: str) -> str:
    # Tokenize inputs
    inputs = tokenizer(
        question,
        context,
        add_special_tokens=True,
        return_tensors="pt"
    )

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

    # Get best start/end token positions
    start_idx = torch.argmax(start_scores)
    end_idx = torch.argmax(end_scores) + 1

    # Convert tokens back to string
    answer_tokens = inputs["input_ids"][0][start_idx:end_idx]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

    # Handle SQuAD v2 "no answer" case
    no_answer_score = start_scores[0][0] + end_scores[0][0]
    best_span_score = start_scores[0][start_idx] + end_scores[0][end_idx-1]
    if no_answer_score > best_span_score:
        return "No answer found"

    return answer
