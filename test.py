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


def group_expand(key_states, layer_groups, num_groups, tgt_size, device='cpu'):
    key_states_expanded = torch.zeros(tgt_size[0], tgt_size[2], tgt_size[1], tgt_size[3]).to(device)
    for i in range(num_groups):
        key_states_expanded[:, :, layer_groups[i], :] = key_states[:, :, i:i+1, :].expand(tgt_size[0], tgt_size[2], len(layer_groups[i]), tgt_size[-1])
    return key_states_expanded.transpose(1, 2)



device = 'cuda'
model_name = "Cheng98/llama-160m"
torch.manual_seed(42)
bsz = 1
q_len = 2
num_heads = 6
head_dim = 4
layer_id = 0

groups_idx = [[[0, 5, 3], [2, 4], [1]]] * 12
num_groups = len(groups_idx[layer_id])

query_states = torch.randn(*(bsz, num_heads, q_len, head_dim), device=device)
key_states = torch.randn(*(bsz, q_len, num_groups, head_dim), device=device)
key_states = group_expand(key_states, groups_idx[layer_id], num_groups, query_states.size(), device=key_states.device)

#model = LlamaForCausalLM.from_pretrained(model_name).to(device)

#model.config.groups_idx = groups_idx
#gqa_model = modeling_llama_gqa.LlamaForCausalLM(model.config).to(device)
#gqa_model_random = modeling_llama_gqa.LlamaForCausalLM(model.config).to(device)
#state = model.state_dict()
#gqa_model.load_state_dict(mha2gqa(state, groups_idx, num_heads=12, transpose_layer=True))


'''
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
generate_ids_mqa_random = my_mqa_model_random.generate(inputs.input_ids, max_length=40)
generate_ids_mqa_transpose = my_mqa_model_transpose.generate(inputs.input_ids, max_length=40)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0], end="\n")
print(tokenizer.batch_decode(generate_ids_mqa, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0], end="\n")
print(tokenizer.batch_decode(generate_ids_mqa_random, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0], end="\n")
print(tokenizer.batch_decode(generate_ids_mqa_transpose, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0], end="\n")
'''