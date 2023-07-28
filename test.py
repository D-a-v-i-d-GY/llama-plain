from transformers.models.llama import LlamaTokenizer, LlamaForCausalLM
from transformers import pipeline
import torch
import numpy as np
import os
import modeling_llama_mqa
import modeling_llama
from mqa import mha2mqa
from data_module import MyDataModule
from datasets import load_dataset
from torch.utils.data import DataLoader


model_name = "Cheng98/llama-160m"


model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name)
my_mqa_model = modeling_llama_mqa.LlamaForCausalLM(model.config)
my_mqa_model_random = modeling_llama_mqa.LlamaForCausalLM(model.config)
my_mqa_model_transpose = modeling_llama_mqa.LlamaForCausalLM(model.config)

state = model.state_dict()
state = mha2mqa(state, num_layers=12, num_heads=12)
my_mqa_model_transpose.load_state_dict(state)

state = model.state_dict()
state = mha2mqa(state, num_layers=12, num_heads=12, transpose_layer=False)
my_mqa_model.load_state_dict(state)

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=40)
generate_ids_mqa = my_mqa_model.generate(inputs.input_ids, max_length=40)
generate_ids_mqa_random = my_mqa_model.generate(inputs.input_ids, max_length=40)
generate_ids_mqa_transpose = my_mqa_model.generate(inputs.input_ids, max_length=40)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0], end="\n")
print(tokenizer.batch_decode(generate_ids_mqa, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0], end="\n")
print(tokenizer.batch_decode(generate_ids_mqa_random, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0], end="\n")
print(tokenizer.batch_decode(generate_ids_mqa_transpose, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0], end="\n")
