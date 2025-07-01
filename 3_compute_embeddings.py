import os
import torch
import numpy as np
import re
import polars as pl
from transformers import AutoTokenizer, AutoModel


# -----------------------------
# --- PARAMETERS
# -----------------------------

INPUT_FOLDER = \
    "../swissubase_2579_1_0/data/wns_corpus_v1.0.0/data/corpus_llm_ready/"
OUTPUT_FOLDER = \
    "../swissubase_2579_1_0/data/wns_corpus_v1.0.0/data/corpus_embeddings/"
    
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

BATCH_SIZE = 4
STARTING_ID = 0
N_BATCHES = 3

# -----------------------------
# --- CODE 
# -----------------------------

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# List files
all_text_files = os.listdir(INPUT_FOLDER)
all_text_files.sort()

todo_text_files = all_text_files[STARTING_ID:(STARTING_ID + N_BATCHES * BATCH_SIZE)]

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Load the tokenizer and model
model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)

# Loop on batches
for batch_id in range(N_BATCHES):
    
    # Print the batch ID
    print(f"Processing batch {batch_id + 1} of {N_BATCHES}")
    
    # Empty the cache 
    torch.cuda.empty_cache()

    # Get the text files for the current batch
    text_files = todo_text_files[(batch_id * BATCH_SIZE):
        (batch_id * BATCH_SIZE + BATCH_SIZE)]

    # Load the files 
    texts = []
    for file_name in text_files:
        file_path = os.path.join(INPUT_FOLDER, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            texts.append(f.read())

    # Tokenize the input texts
    batch_dict = tokenizer(
        texts,
        padding=True,
        return_tensors="pt"
    )

    # Embed the tokens
    batch_dict.to(model.device)
    with torch.no_grad():
        outputs = model(**batch_dict)
    embeddings = outputs.last_hidden_state.to(device="cpu").detach()

    # Loop on files
    for text_id, text_file in enumerate(text_files):
        
        # Get the input IDs for the current text
        input_id = batch_dict["input_ids"][text_id]
        
        # Decode the messages
        tokens = [tokenizer.decode(token_id) for token_id in input_id]

        # Get the markers for the messages
        markers = [{"begin_msg": re.search(r":", token) is not None,
                    "end_msg": re.search(r"(\n|endoftext)", token) is not None}
                for token in tokens]

        # Make the token message groups vector
        token_msg_grps = []
        head_n_markers = 0
        msg_id = 1
        for marker in markers:
            if head_n_markers < 3:
                if marker["begin_msg"]:
                    head_n_markers += 1
                token_msg_grps.append(0)
            else:
                token_msg_grps.append(msg_id)
                if marker["end_msg"]:
                    head_n_markers = 0
                    msg_id += 1
        
        # Ge the number of messages
        n_msg = max(token_msg_grps)

        # Put them in a tensor and to the model device
        token_msg_grps = torch.tensor(token_msg_grps, dtype=torch.int32)

        # Make the message embeddings
        msg_embeddings = torch.zeros((n_msg, embeddings.shape[2]), 
                                    dtype=torch.float16)
        for msg_id in range(1, n_msg + 1):
            # Get the embeddings for the current message
            msg_embedding = embeddings[text_id, 
                                    token_msg_grps==msg_id, :].mean(dim=0)
            # Store the embeddding
            msg_embeddings[msg_id-1, :] = msg_embedding
        
        # Save the embeddings
        short_name = text_file.split(".")[0]
        result_file = os.path.join(OUTPUT_FOLDER, f"{short_name}.csv")
        pl.DataFrame(msg_embeddings).write_csv(result_file, include_header=False)
    
    # Print completion of the batch
    print(f"Batch {batch_id + 1} completed ({", ".join(text_files)})")