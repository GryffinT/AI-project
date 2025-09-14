# dependencies
import os
import random
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # path 
SQUAD_FILE = os.path.join(BASE_DIR, "train-v2.0.json")  # path to dataset
OUTPUT_DIR = os.path.join(BASE_DIR, "qa_generator") # Where to store the trained model/output location
MAX_LENGTH = 512 # max token length
NUM_EPOCHS = 3 # iterations over the datset for training
TRAIN_BATCH_SIZE = 2 # Size of each batch for training
EVAL_BATCH_SIZE = 2 # ^
VALIDATION_SPLIT = 0.1 # 10% of the dataset is set aside for validation
FP16 = True  # set to False if you don't have GPU mixed precision
# ---------------------------------------

# ---------------- LOAD SQUAD ----------------
def load_squad_qa(squad_path): # Load the dataset and turn it into Q: A format
    with open(squad_path, "r", encoding="utf-8") as f: # context messager oppening the dataset with utf-8 encoding
        data = json.load(f)["data"] # store it in data

    qa_list = [] # Question Answer list 
    for article in data: # for each entry within the data stored in the SQuAD dict
        for para in article["paragraphs"]: # for each paragraph in the data
            for qa in para["qas"]: # for every Q/A pair per paragraph
                question = qa["question"].strip() # strip the question (remove spaces)
                if not qa["answers"] or qa.get("is_impossible", False): # If theres no answer or the question is impossible
                    answer = "I'm not sure, based on the data provided I am unable to answer that." # set an arbitrary message to map to each unanswerable question
                else: # If there is an answer
                    answer = qa["answers"][0]["text"].strip() # set the stripped answer to the answer variable
                qa_list.append(f"Q: {question} A: {answer}") # Add the Q and A to the qa_list
    return qa_list # when its done, return the list of all questions mapped to their respective answers

# ----------------- PREPARE DATASET -----------------
qa_pairs = load_squad_qa(SQUAD_FILE) # Function from above, just takes the data and maps each paragraph/question to an answer or no answer if that is the case
random.shuffle(qa_pairs) # shuffle the Q/A pairs 


# Just test train split 
split_idx = int(len(qa_pairs) * (1 - VALIDATION_SPLIT)) 
train_texts = qa_pairs[:split_idx]
valid_texts = qa_pairs[split_idx:]

train_dataset = Dataset.from_dict({"text": train_texts})
valid_dataset = Dataset.from_dict({"text": valid_texts})

# ----------------- TOKENIZER -----------------
tokenizer = AutoTokenizer.from_pretrained("gpt2") # tokenizer initialized
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token so we use EOS (end of statement) token to equalize lengths of tokens <EOS>

def tokenize(batch): # batch tokenization function
    tokens = tokenizer( # tokenizer obj and args
        batch["text"], # batching
        truncation=True, # cut down the tokens if they're over max length
        max_length=MAX_LENGTH, # max length
        padding="max_length" # pad the tokens with <EOS> if they're under max length
    )
    tokens["labels"] = tokens["input_ids"].copy() # add all the tokens to the labels list, masking 
    return tokens

train_tokenized = train_dataset.map(tokenize, batched=True, remove_columns=["text"]) # tokenized training datasaet
valid_tokenized = valid_dataset.map(tokenize, batched=True, remove_columns=["text"]) # tokenized validation dataset

# ----------------- TRAINING ARGUMENTS -----------------
training_args = TrainingArguments( # Arguments for the training loop
    output_dir=OUTPUT_DIR, # output path
    num_train_epochs=NUM_EPOCHS, # training iterations over the dataset
    per_device_train_batch_size=TRAIN_BATCH_SIZE, # batch size per CPU/GPU
    per_device_eval_batch_size=EVAL_BATCH_SIZE, # ^
    save_strategy="epoch", # save the model after each epoch
    eval_strategy="epoch", # eval model per epoch
    logging_dir=os.path.join(BASE_DIR, "logs"), # output model logs to model
    logging_steps=50, # log each 50 steps
    save_total_limit=2, # save limit
    fp16=FP16, # mixed precision - floating point 16
)

# ----------------- MODEL -----------------
model = AutoModelForCausalLM.from_pretrained("gpt2") # initialize model 

trainer = Trainer( # training loop with specified args
    model=model, 
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=valid_tokenized,
    tokenizer=tokenizer
)

# ----------------- TRAIN -----------------
trainer.train() # begin training loop

# ----------------- SAVE -----------------
trainer.save_model(OUTPUT_DIR) 
tokenizer.save_pretrained(OUTPUT_DIR)
