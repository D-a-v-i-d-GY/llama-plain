from datasets import load_dataset
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.tokenization_llama import LlamaTokenizer
import torch
from tqdm import tqdm
import modeling_llama_mqa
import modeling_llama_gqa
from architecture_transform_util import mha2mqa, mha2gqa


def calculate_ppl(model, encodings, device, stride=512, max_length=1024):
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl


device = 'cuda'
model_name = "Cheng98/llama-160m"
torch.manual_seed(420)

tokenizer = LlamaTokenizer.from_pretrained(model_name)

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt").to(device)

groups_idx = [[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]]] * 12

model = LlamaForCausalLM.from_pretrained(model_name).to(device)
model_random = LlamaForCausalLM(model.config).to(device)
mqa_model = modeling_llama_mqa.LlamaForCausalLM(model.config).to(device)
mqa_model_random = modeling_llama_mqa.LlamaForCausalLM(model.config).to(device)

model.config.groups_idx = groups_idx
gqa_model = modeling_llama_gqa.LlamaForCausalLM(model.config).to(device)
gqa_model_random = modeling_llama_gqa.LlamaForCausalLM(model.config).to(device)

state = model.state_dict()
gqa_model.load_state_dict(mha2gqa(state, groups_idx, num_heads=12, transpose_layer=True))

state = model.state_dict()
mqa_model.load_state_dict(mha2mqa(state, num_layers=12, num_heads=12, transpose_layer=False))


with torch.inference_mode():
    model.eval()
    model_random.eval()
    mqa_model.eval()
    mqa_model_random.eval()
    gqa_model.eval()
    gqa_model_random.eval()

    ppl = calculate_ppl(model, encodings, device)
    ppl_random = calculate_ppl(model_random, encodings, device)
    ppl_mqa = calculate_ppl(mqa_model, encodings, device)
    ppl_mqa_random = calculate_ppl(mqa_model_random, encodings, device)
    ppl_gqa = calculate_ppl(gqa_model, encodings, device)
    ppl_gqa_random = calculate_ppl(gqa_model_random, encodings, device)


print("base: ", ppl)
print("base model, random weights: ", ppl_random)
print("MHA -> MQA, transformed weights: ", ppl_mqa)
print("MHA -> MQA, random weights: ", ppl_mqa_random)
print(f"MHA -> GQA with {len(groups_idx[0])} groups, transformed weights: ", ppl_gqa)
print(f"MHA -> GQA with {len(groups_idx[0])} groups, random weights: ", ppl_gqa_random, end="\n")