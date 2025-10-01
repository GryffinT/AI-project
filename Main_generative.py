from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Load trained model + tokenizer
model_name = "./my_squad_trained_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

def answer_question(question, context):
    inputs = tokenizer(
        question,
        context,
        add_special_tokens=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

    # Get the most likely start and end tokens
    start_idx = torch.argmax(start_scores)
    end_idx = torch.argmax(end_scores) + 1

    # Decode the tokens back to string
    answer_tokens = inputs["input_ids"][0][start_idx:end_idx]
    answer = tokenizer.decode(answer_tokens)

    # Handle "no answer" case (SQuAD v2)
    no_answer_score = outputs.start_logits[0][0] + outputs.end_logits[0][0]
    if no_answer_score > (start_scores[0][start_idx] + end_scores[0][end_idx-1]):
        return "No answer found"

    return answer
