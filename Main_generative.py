# Dependencies
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Load tokenizer and final model from the repo
tokenizer = AutoTokenizer.from_pretrained("GryffinT/SQuAD.QA")
model = AutoModelForQuestionAnswering.from_pretrained("GryffinT/SQuAD.QA")

# Set up the QA pipeline
qa_pipeline = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer,
    device=0  # set to -1 if CPU only
)

# Function for UI to call
def output(question, context):
    """
    Takes a question + context string and returns the best answer.
    """
    result = qa_pipeline(question=question, context=context)
    return result["answer"]

