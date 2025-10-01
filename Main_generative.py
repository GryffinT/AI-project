# Dependencies
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load tokenizer and final model from the repo
tokenizer = AutoTokenizer.from_pretrained("GryffinT/SQuAD.QA") # Edit the first dir for repo access, then the subfolder if and ONLY if the model is in a folder WITHIN the repo.
model = AutoModelForCausalLM.from_pretrained("GryffinT/SQuAD.QA")
# If you have a GPU, use device=0; otherwise omit it for CPU
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0  # remove or set to -1 if no GPU
)

# Function to generate text
def output(prompt):
    out = generator(
        prompt,
        max_new_tokens=600,     # how many tokens to generate
        do_sample=True,         # sample instead of greedy
        temperature=0.9,        # randomness of sampling
        top_p=0.95,             # nucleus sampling
        top_k=50,               # top-k sampling
        num_return_sequences=1, # generate 1 response
        repetition_penalty=1.1, # discourage repeating words
        return_full_text=True   # include the prompt in output
    )
    return out[0]["generated_text"]


