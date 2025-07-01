import os
import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


# -----------------------------
# --- PARAMETERS
# -----------------------------

FILE_PATH = \
    "../swissubase_2579_1_0/data/wns_corpus_v1.0.0/data/corpus_llm_ready/wns_chat_30.txt"
RESULT_FOLDER = \
    "../swissubase_2579_1_0/data/wns_corpus_v1.0.0/data/corpus_embeddings/"
    
MODEL_NAME = "Qwen/Qwen3-Embedding-8B"

# -----------------------------
# --- CODE 
# -----------------------------

with open(FILE_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()


tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B')
model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B', 
                                  torch_dtype=torch.float16)

max_length = 100
 
# Tokenize the input texts
batch_dict = tokenizer(
    input_texts,
    padding=True,
    truncation=True,
    max_length=max_length,
    return_tensors="pt"
)
batch_dict.to(model.device)
outputs = model(**batch_dict)