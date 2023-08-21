from transformers.models.llama import LlamaTokenizer, LlamaForCausalLM
from transformers import pipeline
import torch
import numpy as np
import os
import modeling_llama_gqa
import modeling_llama_mqa
import modeling_llama
from architecture_transform_util import mha2mqa, mha2gqa
from data_module import MyDataModule
from datasets import load_dataset
from torch.utils.data import DataLoader

def merge_list(input_list):
    out = []
    for sublist in input_list:
        out += sublist

    return out


device = 'cuda'
model_name = "Cheng98/llama-160m"
torch.manual_seed(42)

model = LlamaForCausalLM.from_pretrained(model_name).to(device)
tokenizer = LlamaTokenizer.from_pretrained(model_name)
test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
print(type(test["text"]))
encodings = tokenizer(test["text"], return_tensors="pt")
print(type(encodings))
encodings = merge_list(encodings["input_ids"][:100])
encodings = merge_list(encodings)
encodings = torch.tensor(encodings).to(device)

encodings1 = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

print(encodings1["input_ids"].shape)