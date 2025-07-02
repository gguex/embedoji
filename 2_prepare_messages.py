import polars as pl
import os
from transformers import AutoTokenizer
import re
import numpy as np

# -----------------------------
# --- PARAMETERS
# -----------------------------

CONTEXT_LENGTH = 32000
INPUT_FILES_FOLDER = \
    "../swissubase_2579_1_0/data/wns_corpus_v1.0.0/data/corpus_csv/"

PSEUDO_PATH = "data/aux/pseudo.csv"
SPECIAL_TAGS_PATH = "data/aux/special_tags.csv"

MODEL_NAME = "Qwen/Qwen3-Embedding-8B"

OUTPUT_FOLDER = \
    "../swissubase_2579_1_0/data/wns_corpus_v1.0.0/data/corpus_llm_ready/"
META_OUTPUT_FOLDER = "data/metadata/llm_ready_metadata/"

# -----------------------------
# --- CODE
# -----------------------------

# External data 
pseudo_df = pl.read_csv(PSEUDO_PATH)
special_tags_df = pl.read_csv(SPECIAL_TAGS_PATH)

# Get the special tags
special_tags = special_tags_df.select(pl.col("tag")).to_series().to_list()

# Get file names
file_names = os.listdir(INPUT_FILES_FOLDER)
file_names.sort()

# Window length and stride
win_length = CONTEXT_LENGTH // 3

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Loop on them 
for file_name in file_names:
        
    short_name = file_name.split(".")[0]
    df = pl.read_csv(os.path.join(INPUT_FILES_FOLDER, file_name))

    # --- Preprocessing messages

    # Remove messages with no user
    df = df.filter(pl.col("msg_user").is_not_null())
    # Remove messages with no text
    df = df.filter(pl.col("msg_text") != "")
    # Remove messages with only special tags
    df = df.filter(~pl.col("msg_text").is_in(special_tags))

    # Convert the DataFrame to a full text file
    msgs = []
    msg_idx = []
    for row in df.iter_rows(named=True):
        msg_text = row["msg_text"].replace("\n", " ")
        msg_date = row["msg_date"]
        if msg_date.count(":") == 2:
            msg_date = ":".join(msg_date.split(":")[:-1])
        msgs.append(f"[{msg_date}] {row['msg_user']}: {msg_text}")
        msg_idx.append(row["msg_id"])
    # Join the messages into a single string
    full_text = "\n".join(msgs)

    # Change the user names to the pseudo names
    for row in pseudo_df.iter_rows(named=True):
        user_id = row["user_id"]
        std_user_id = f"_{'_'.join(user_id.upper().split('.'))}_"
        full_text = full_text.replace(std_user_id, row["pseudo"])
        
    # --- Build context files

    # Get the number of tokens
    number_of_tokens = len(tokenizer(full_text)["input_ids"])

    # If size is ok : write the conversation in a single file
    if number_of_tokens < CONTEXT_LENGTH:
        
        # Write the metadata
        metadata_df = pl.DataFrame({
            "msg_id": msg_idx,
            "context_groups": 0,
        })
        metadata_df.write_csv(os.path.join(META_OUTPUT_FOLDER, 
                                           f"meta_{file_name}"))
        
        # Write the full text
        with open(os.path.join(OUTPUT_FOLDER, f"{short_name}.txt"), 
                "w", encoding="utf-8") as f:
            f.write(full_text)
            
    # If size is NOT ok : split the conversation into multiple contexts
    else:

        # Make the text lengths and cumulative lengths vectors
        split_text = full_text.split("\n")
        msg_lengths = []
        cum_lengths = []
        cum_length = 0
        for msg in split_text:
            msg_length = len(tokenizer(msg + "\n")["input_ids"])
            msg_lengths.append(msg_length)
            cum_length += msg_length
            cum_lengths.append(cum_length)
        
        std_length = win_length - np.max(msg_lengths)
        n_contexts = int(cum_lengths[-1] / std_length) + 1
        token_size = cum_lengths[-1] / n_contexts

        # Compute the context groups with nearest cut
        context_groups = np.zeros(len(cum_lengths))
        for i in range(1, n_contexts):
            threshold = i * token_size
            diffs = np.abs(np.array(cum_lengths) - threshold)
            closest_id = np.where(diffs == np.min(diffs))[0][0].item()
            context_groups[closest_id:] += 1
            
        # Make the contexts
        contexts = []
        for i in range(n_contexts):
            context = "\n".join(np.array(split_text)[context_groups == i])
            contexts.append(context)

        # Save metadata
        metadata_df = pl.DataFrame({
            "msg_id": msg_idx,
            "context_groups": context_groups.astype(np.int32).tolist(),
        })
        metadata_df.write_csv(os.path.join(META_OUTPUT_FOLDER, 
                                           f"meta_{file_name}"))

        # Save the contexts
        for i in range(len(contexts)):
            if not (i == 0 or i == (len(contexts) - 1)):
                full_context = contexts[i-1] + "\n" + contexts[i] + "\n" \
                    + contexts[i+1]
                c_str = f"{i-1}-{i}-{i+1}"
                with open(os.path.join(OUTPUT_FOLDER, 
                                       f"{short_name}_{c_str}.txt"), 
                        "w", encoding="utf-8") as f:
                    f.write(full_context)
                    