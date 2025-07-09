import os
import polars as pl
import polars.selectors as cs
from sklearn.metrics.pairwise import cosine_similarity
from src.mv_tools import *
import plotly.express as px
from nicegui import ui, run

# -----------------------------
# --- PARAMETERS
# -----------------------------

EMBEDDINGS_FOLDER = "data/data/corpus_embeddings/"
CORPUS_CSV_FOLDER = \
    "../swissubase_2579_1_0/data/wns_corpus_v1.0.0/data/corpus_csv/"
EMBEDDINGS_META_FOLDER = "data/metadata/llm_ready_metadata/"
EMBEDDINGS_SIZE = 1024
    
DATE_SPAN = [pl.date(2020, 3, 16) , pl.date(2020, 6, 20) ]

DIM_MAX = 100

OUTPUT_FOLDER = "data/data/embeddings_mds/"

# -----------------------------
# --- CODE 
# -----------------------------

# ---- Construct the metadata for the embeddings

# Get the chat names
chat_names =  os.listdir(CORPUS_CSV_FOLDER)
chat_names.sort()

# Get the csv of messages
csv_df = pl.DataFrame(None, schema= [("msg_id", pl.Int64), 
                       ("chat_name", pl.String), 
                       ("msg_date", pl.Datetime), 
                       ("msg_user", pl.String), 
                       ("msg_text", pl.String), 
                       ("emoji_list", pl.String)])
for chat_name in chat_names:
    chat_df = pl.read_csv(CORPUS_CSV_FOLDER + chat_name, 
                          try_parse_dates=True, infer_schema_length=1000)
    csv_df.extend(chat_df)
    
# Get the metadata of embeddings
emb_meta_df = pl.scan_csv(EMBEDDINGS_META_FOLDER + "meta_wns_chat_*.csv", 
                      include_file_paths="path").collect()
# Add the file name to the metadata
emb_meta_df = emb_meta_df.with_columns(
    pl.col("path").str.split(by="meta_").list.last().alias("file_name")
).drop("path")
# Add the the context to the file name 
emb_meta_df = emb_meta_df.with_columns(
    (pl.concat_str([
        pl.col("file_name").str.replace_all(".csv", ""),
        pl.col("context_groups"),
    ], separator="_") + ".csv").alias("context_name")
)

# Select the pertinent messages
sel_csv_df = csv_df.filter((pl.col("msg_date").is_between(*DATE_SPAN)))

# Inner join the csv with the metadata
meta_df = sel_csv_df.join(emb_meta_df, 
                          left_on="msg_id", right_on="msg_id", how="inner")

# ---- Get the embeddings

# Get the pertinent embeddings files
context_file_names = list(set(meta_df["context_name"].to_list()))
context_file_names.sort()

# Get the embeddings
embeddings = np.zeros((meta_df.shape[0], EMBEDDINGS_SIZE), dtype=np.float32)
for context_file_name in context_file_names:
    emb_meta = meta_df.with_row_index().filter(
        pl.col("context_name") == context_file_name)
    msg_ids = emb_meta["msg_id"].to_list()
    emb_ids = emb_meta["index"].to_numpy()
    emb_df = pl.read_csv(EMBEDDINGS_FOLDER + context_file_name, 
                         has_header=False)
    embeddings[emb_ids, :] = emb_df.filter(
        pl.col("column_1").is_in(msg_ids)).drop("column_1").to_numpy()

# Save the number of embeddings
n_embeddings = embeddings.shape[0]
    
# ------------------------------------------
# ---- Make the MDS
# ------------------------------------------

# ---- MDS on messages

# Weights
weights = np.ones([n_embeddings])
weights /= np.sum(weights)

# Compute the cosine similarity
cosine_sim = cosine_similarity(embeddings)

# Compute it
wdt_scalarp_mat = weighted_scalar_product_matrix(cosine_sim, weights)
X_mds, eigenvalues = weighted_mds(wdt_scalarp_mat, weights=weights, 
                                  dim_max=DIM_MAX)

# For the first plot
mds_df = pl.from_numpy(X_mds).rename(lambda c_n : f"MDS_{c_n.split("_")[1]}")
results_df = meta_df.with_columns(mds_df)

# Save the results
results_df.write_csv("data/data/embeddings_mds/embeddings_mds.csv")