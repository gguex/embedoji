import polars as pl
import os
from transformers import AutoTokenizer
import re
import numpy as np

# -----------------------------
# --- PARAMETERS
# -----------------------------

CONTEXT_LENGTH = 32000
INPUT_FILE_FOLDER = "../swissubase_2579_1_0/data/wns_corpus_v1.0.0/data/corpus_csv/"

USR_PSEUDO_PATH = "data/pseudo.csv"
SPECIAL_TAG_PATH = "data/special_tags.csv"

TOKENIZER_NAME = "Qwen/Qwen3-Embedding-8B"

OUTPUT_FOLDER = "../swissubase_2579_1_0/data/wns_corpus_v1.0.0/data/corpus_llm_ready/"
OUTPUT_META_FOLDER = "../swissubase_2579_1_0/data/wns_corpus_v1.0.0/data/corpus_llm_ready_metadata/"

# -----------------------------
# --- CODE
# -----------------------------

# External data 
pseudo_df = pl.read_csv(USR_PSEUDO_PATH)
special_tags_df = pl.read_csv(SPECIAL_TAG_PATH)

# Get the special tags
special_tags = special_tags_df.select(pl.col("tag")).to_series().to_list()

# Get file names
file_names = os.listdir(INPUT_FILE_FOLDER)
file_names.sort()

# Loop on them 
for file_name in file_names :
        
    short_name = file_name.split(".")[0]
    df = pl.read_csv(os.path.join(INPUT_FILE_FOLDER, file_name))

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
        msgs.append(f"[{row["msg_date"]}] {row["msg_user"]}: {row["msg_text"]}")
        msg_idx.append(row["msg_id"])
    # Join the messages into a single string
    full_text = "\n".join(msgs)

    # Change the user names to the pseudo names
    for row in pseudo_df.iter_rows(named=True):
        user_id = row["user_id"]
        std_user_id = f"_{"_".join(user_id.upper().split("."))}_"
        full_text = full_text.replace(std_user_id, row["pseudo"])
        
    # --- Build context files

    # Windows length
    win_length = CONTEXT_LENGTH // 3
        
    # The tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Get the number of tokens
    number_of_tokens = len(tokenizer(full_text)["input_ids"])

    # If size ok : write the whole file
    if number_of_tokens < CONTEXT_LENGTH:
        
        # Write the metadata
        metadata_df = pl.DataFrame({
            "msg_id": msg_idx,
            "context_groups": 0,
        })
        metadata_df.write_csv(os.path.join(OUTPUT_META_FOLDER, 
                                           f"meta_{file_name}"))
        
        # Write the full text
        with open(os.path.join(OUTPUT_FOLDER, f"{short_name}.txt"), 
                "w", encoding="utf-8") as f:
            f.write(full_text)
            
    # If not : split the text into contexts
    else:
        # Compute the number contexts needed
        number_of_contexts = (number_of_tokens // win_length) + 1

        # Make the cumulative text lengths vector
        split_text = full_text.split("\n")
        cum_lengths = []
        cum_length = 0
        for msg in split_text:
            msg_length = len(tokenizer(msg)["input_ids"])
            cum_length += msg_length
            cum_lengths.append(cum_length)

        # Compute the context groups from quantiles
        context_groups = number_of_contexts - np.zeros(len(cum_lengths))
        for i in range(1, number_of_contexts + 1):
            quantile = np.quantile(cum_lengths, i/number_of_contexts)
            context_groups -= 1* (cum_lengths <= quantile)
            
        # Make the contexts
        contexts = []
        for i in range(number_of_contexts):
            context = "\n".join(np.array(split_text)[context_groups == i])
            contexts.append(context)

        # Save metadata
        metadata_df = pl.DataFrame({
            "msg_id": msg_idx,
            "context_groups": context_groups.astype(np.int32).tolist(),
        })
        metadata_df.write_csv(os.path.join(OUTPUT_META_FOLDER, 
                                           f"meta_{file_name}"))

        # Save the contexts
        for i in range(len(contexts)):
            if not (i == 0 or i == (len(contexts) - 1)):
                full_context = contexts[i-1] + "\n" + contexts[i] + "\n" \
                    + contexts[i+1]
                c_str = f"{i-1}-{i}-{i+1}"
                with open(os.path.join(OUTPUT_FOLDER, f"{short_name}_{c_str}.txt"), 
                        "w", encoding="utf-8") as f:
                    f.write(full_context)
                    