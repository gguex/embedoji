import os
import numpy as np
import re
import polars as pl

# -----------------------------
# --- PARAMETERS
# -----------------------------

INPUT_FOLDER = \
    "../swissubase_2579_1_0/data/wns_corpus_v1.0.0/data/corpus_embeddings/"
INPUT_META_FOLDER = "data/metadata/llm_ready_metadata/"

OUTPUT_FOLDER = "data/data/corpus_embeddings"

# -----------------------------
# --- CODE 
# -----------------------------

# File counters
file_counter = 0

# List files
all_embedding_files = os.listdir(INPUT_FOLDER)
all_embedding_files.sort()

# Loop on files
for file_name in all_embedding_files:

    # Get the chat
    chat_name = re.match(r"wns_chat_\d\d", file_name).group(0)

    # Load the embeddings
    embeddings_df = pl.read_csv(os.path.join(INPUT_FOLDER, file_name), 
                                has_header=False)
    # Get the metadata file
    meta_df = pl.read_csv(os.path.join(INPUT_META_FOLDER, f"meta_{chat_name}.csv"))

    # Get the subgroups
    subchats_str = file_name.split(chat_name)[1].split(".")[0][1:]
    if subchats_str == "":
        subchats_str = "0"
    subchats = [int(subchat) for subchat in subchats_str.split("-")]
    # Filter the metadata DataFrame
    filt_meta_df = meta_df.filter(pl.col("context_groups").is_in(subchats))

    # Check if the file are ok
    if filt_meta_df.shape[0] != embeddings_df.shape[0]:
        raise ValueError(f"File {file_name} has not the right number of rows")

    # Get the last chat ID
    last_chat = meta_df["context_groups"].max()
    # Make the potential list of subchats to treat
    potential_todo_subchats = set([0] + subchats[1:-1] + [last_chat])
    # Make the list of subchats to treat
    todo_subchats = list(set(subchats).intersection(potential_todo_subchats))


    # Write the embeddings
    for subchat_id in todo_subchats:
        
        # Metadata selection
        kept_indices = filt_meta_df["context_groups"] == subchat_id
        kept_emb_df = embeddings_df.filter(kept_indices)
        kept_msg_ids = pl.DataFrame(filt_meta_df.filter(kept_indices)["msg_id"])

        # Make the final df
        emb_out_df = kept_msg_ids.with_columns(kept_emb_df)
        
        # Write the final df
        output_path = os.path.join(OUTPUT_FOLDER, f"{chat_name}_{subchat_id}.csv")
        emb_out_df.write_csv(output_path, include_header=False)
        file_counter += 1
    
# Print the result
print(f"Processed {file_counter} files")