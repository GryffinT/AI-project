# Dependencies

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import pipeline


dataset = load_dataset("text", data_files={"train": "training_text.txt", "validation": "validation_text.txt"}) # takes the training text and validation texts and aggregates a dictionary within data. "training"/"validation" = keys

tokenizer = AutoTokenizer.from_pretrained("gpt2") # sets the tokenizer to the pretrained transformer's tokenizer
tokenizer.pad_token = tokenizer.eos_token # initializes the padding token to use as the... EOS token cause gpt dosent have one nativley... not reccommended unless you use masking.
# For refference, padding turns ["I", "love", "my", "dog"] into ["I", "love", "my", "dog", "<EOS>"], it also turns ["Hello", "World"] into ["Hello", "World", "<EOS>", "<EOS>", "<EOS>"]. it just normalizes the lengths
# Also, calling this like tokenizer(string) would return a dictionary where {string: list[ints]}, that might look like:
#tokenizer("I love dogs") = {
#   input_token_ids:[101, 1045, 2293, 3678, 102]],
#   attention_mask: [1, 1, 1, 1, 1] # Because theres no padding theres no zeros in the attention_mask list!
#} 

def tokenize(batch): # Define the tokenize function that takes in batches
  tokens = tokenizer(batch["text"], # tokens dictionary equals batches of text passed through the tokenizer ^ 
                     truncation = True, # if the batch is longer than 512, remove the excess.
                     max_length = 512, # the longest allowed batch
                     padding = "max_length") # if the example is under 512 pad it until it reaches 512 ^
  tokens["labels"] = tokens["input_ids"].copy() # create a label entry in the tokens dict and populate it's values with a list of the input_ids, tokens["input_ids"] = tokens["labels"]
  return(tokens) # return the finalized tokens dict.

tokenized = dataset.map(tokenize, batched = True, remove_columns = ["text"]) # Fancy high level one liner.
# Creates a dictionary. dataset = the loaded dataset from earlier, .map = fancy high-level iterator that usess paralel batch iteration through the dataset, tokenized is the function right above, batched = use batching = True.
# And remove_columns = ["text"] just means to remove the leftover "text" column from before tokenization, pretty much just data cleanup.

args = TrainingArguments( # Arguments for the training loop, how it should behave.
  output_dir="out", # Output directory, where should it send its logs and stuff to?
  num_train_epochs=3, # number of times it's gonna pass over the training set.
  per_device_train_batch_size=2, # training batch size per CPU/GPU.
  per_device_eval_batch_size=2, # evaluation/validation batch size per CPU/GPU.
  save_strategy="epoch", # save the model and its training progress each epoch.
  evaluation_strategy="epoch", # evaluate the model's training progress each epoch.
  logging_steps=50, # Log it's progress each 50 steps.
  fp16=True # use mixed precision (float16) if possible.
)

trainer = Trainer(
  model=model = AutoModelForCausalLM.from_pretrained("gpt2"), # gets the transformer model.
  args = args, # sets the training loop's arguments to the ones set above.
  train_dataset = tokenized["train"], # sets the training dataset to the training examples from the tokenized dataset.
  eval_dataset = tokenized["validation"], #  ^ 
  tokenizer = tokenizer # sets the loop's tokenizer to our predefined tokenizer".
)

trainer.train() # runs the training loop
trainer.save_model("text_generator") # saves the model's progress wthin the text generator file.
tokenizer.save_pretrained("text_generator") # saves the model's config and vocab to the text generator file aswell. 

generator = pipeline("text-generation", model="text_generator", tokenizer="text_generator", device=0) # establishes the transformer pipeline in text generation oode using the text_generator model and tokenizer on the GPU.

def output(prompt):
  out = generator( # output of the pipeline =.
    prompt, # Input prompt.
    max_new_tokens = 600, # The maximum length of the generated response in tokens NOT INCLUDING THE PROMPT IF return_full_text == True.
    do_sample = True, # use sampling, this means it samples the data with variance in predicted word choice rather than just the most likely, give natural variance to responses.
    temperature = 0.9, # Temperature == randomness of sampling, 0 - 1, 1 being random 0 being conservative.
    top_p = 0.95, # nucleus sampling, chooses the smallest set of tokens with a cumulative probability >= 95% and random samples it following do_sample and temp
    top_k = 50, # Limites the sampling to the top 50 most probable choices.
    num_return_sequences = 1, # Number of responses to generate.
    repetition_penalty = 1.1, # punishes the model for repeating phrases/terms to ensure a nice natural output.
    return_full_text = True, # Returns the generated response AND the initial prompt.
  )
  return out[0]["generated_text"]


