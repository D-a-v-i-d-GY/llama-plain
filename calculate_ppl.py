from datasets import load_dataset
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.tokenization_llama import LlamaTokenizer
import torch
from tqdm import tqdm
import modeling_llama_mqa
from mqa import mha2mqa


def calculate_ppl(model, encodings, device, stride=1024, max_length=1024):
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

model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name)
my_mqa_model = modeling_llama_mqa.LlamaForCausalLM(model.config)
my_mqa_model_random = modeling_llama_mqa.LlamaForCausalLM(model.config).to(device)

state = model.state_dict()
state = mha2mqa(state, num_layers=12, num_heads=12, transpose_layer=True)

my_mqa_model.load_state_dict(state)
my_mqa_model.to(device)

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt").to(device)

model = model.to(device)
with torch.inference_mode():
    my_mqa_model.eval()
    my_mqa_model_random.eval()
    ppl = calculate_ppl(model, encodings, device)
    ppl_mqa = calculate_ppl(my_mqa_model, encodings, device)
    ppl_mqa_random = calculate_ppl(my_mqa_model_random, encodings, device)

print("base: ", ppl)
print("averaged: ", ppl_mqa)
print("random: ", ppl_mqa_random)