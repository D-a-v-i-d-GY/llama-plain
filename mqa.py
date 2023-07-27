import torch
from torch import nn, einsum
from typing import Optional

# Use a pipeline as a high-level helper
#from transformers import pipeline
# Load model directly
#from transformers.models.llama import LlamaTokenizer
#from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device=device)

#tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
#model = AutoModelForCausalLM.from_pretrained("Cheng98/llama-160m")

torch.manual_seed(42)

b = 4 # batch size
d = 8 # model dim
m = 4 # number of previous tokens, i.e. trying to predict token number m+1
h = 4 # number of heads
k = v = d // h # dimension of q, k, v heads

x = torch.randn(b, m, d)
P_q = torch.randn(h, d, k)
P_k = torch.randn(d, k)
P_v = torch.randn(d, v)
P_o = torch.randn(h, d, v)

mask = torch.tril(torch.ones(m, m))
mask[mask==0] = -torch.inf
mask[mask==1] = 0

def mqa_logic(x, mask=mask, prev_K=None, prev_V=None, P_q=P_q, P_k=P_k, P_v=P_v, P_o=P_o):
    if prev_K is not None and prev_V is not None:
        inp = x[:, -1, :]
        queries = einsum("bd,hdk->bhk", inp, P_q)
        key = einsum("bd,dk->bk", inp, P_k).unsqueeze(dim=1)
        new_K = torch.cat(prev_K, key, dim=1)
        value = einsum("bd,dv->bv", inp, P_v).unsqueeze(dim=1)
        new_V = torch.cat(prev_V, value, dim=1)

        logits = einsum("bhk,bmk->bhm", queries, new_K)
        weights = nn.functional.softmax(logits, dim=-1)
        o = einsum("bhm,bmv->bhv", weights, new_V)
        y = einsum("bhv,hdv->bd", o, P_o)

    else:
        queries = einsum("bnd,hdk->bhnk", x, P_q)
        new_K = einsum("bmd,dk->bmk", x, P_k)
        new_V = einsum("bmd,dv->bmv", x, P_v)

        logits = einsum("bhnk,bmk->bhnm", queries, new_K)
        weights = nn.functional.softmax(logits + mask, dim=-1)
        o = einsum("bhnm,bmv->bhnv", weights, new_V)
        y = einsum("bhnv,hdv->bnd", o, P_o)
    return y, new_K, new_V

print(mqa_logic(x))