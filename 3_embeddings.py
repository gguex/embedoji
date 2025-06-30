import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel

input_texts = [
    "okay!! super je peux faire des lessives 😛", 
    "<msg_start> moi c'est _MASKED_TEXT_ question 3, je finis de corriger cette semaine",
    "Guillaume"]

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