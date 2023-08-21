from datasets import load_dataset
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.tokenization_llama import LlamaTokenizer
import torch
from tqdm import tqdm
from peft import PeftModel, PeftConfig
import modeling_llama_mqa
import modeling_llama_gqa
from architecture_transform_util import mha2mqa, mha2gqa


def calculate_ppl(model, encodings, stride=512, max_length=2048):
    seq_len = encodings.numel()

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        if max_length == -1: 
            end_loc = seq_len
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings[:, begin_loc:end_loc].to(encodings.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.inference_mode():
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


def merge_list(input_list):
    out = []
    for sublist in input_list:
        out += sublist

    return out


def group_ppl_calc(model, group_idxx):
    ppl_out = []
    for i in range(len(group_idxx)):
        group_idx = group_idxx[i]
        model.config.groups_idx = group_idx
        gqa_model = modeling_llama_gqa.LlamaForCausalLM(model.config).to(device)
        gqa_model_random = modeling_llama_gqa.LlamaForCausalLM(model.config).to(device)

        # transpose_layer should always be True, TESTED
        state = model.state_dict()
        gqa_model.load_state_dict(mha2gqa(state, group_idx, num_heads=12, transpose_layer=True))

        with torch.inference_mode():
            model.eval()
            gqa_model.eval()
            gqa_model_random.eval()

            ppl_gqa = calculate_ppl(gqa_model, encodings, max_length=-1)
            ppl_gqa_random = calculate_ppl(gqa_model_random, encodings, max_length=-1)
        ppl_out[i] = (ppl_gqa, ppl_gqa_random)
    
    return ppl_out

device = 'cuda'
model_name = "Cheng98/llama-160m"
torch.manual_seed(420)
max_length = 1024

# Prepare & encode data
tokenizer = LlamaTokenizer.from_pretrained(model_name)
test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = test.map(lambda x: tokenizer(x["text"], return_tensors="pt"))
encodings = merge_list(encodings["input_ids"])
encodings = merge_list(encodings)
encodings = torch.tensor(encodings).reshape(1, -1).to(device)

# Peft model
model = LlamaForCausalLM.from_pretrained(model_name).to(device)
peft_model_id = "lora_models/plain-lora-0"
model_id = "my-lora-ckpts/checkpoint-4500"
lora_model = PeftConfig.from_pretrained(model_id)
config = PeftConfig.from_pretrained(peft_model_id)
peft_model = PeftModel.from_pretrained(lora_model, peft_model_id).to(device)

# Define groups, very rough implementation #NEEDS IMPROVEMENT
group_idx0 = [[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]] * 12
group_idx1 = [[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]] * 12
group_idx2 = [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]] * 12
group_idx3 = [[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]] * 12
group_idx4 = [[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]]] * 12
group_idx5 = [[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]]] * 12

group_idxx = [group_idx0, group_idx1, group_idx2, group_idx3, group_idx4, group_idx5]


#model_random = LlamaForCausalLM(model.config).to(device)
#mqa_model = modeling_llama_mqa.LlamaForCausalLM(model.config).to(device)
#mqa_model_random = modeling_llama_mqa.LlamaForCausalLM(model.config).to(device)
#state = model.state_dict()
#mqa_model.load_state_dict(mha2mqa(state, num_layers=12, num_heads=12, transpose_layer=True))

with torch.inference_mode():
 #   model_random.eval()
    peft_model.eval()
#    mqa_model.eval()
#    mqa_model_random.eval()

    ppl = calculate_ppl(model, encodings, max_length=max_length)
    ppl_peft = calculate_ppl(peft_model, encodings, max_length=max_length)
 #   ppl_random = calculate_ppl(model_random, encodings, max_length=max_length)
#    ppl_mqa = calculate_ppl(mqa_model, encodings, max_length=max_length)
#    ppl_mqa_random = calculate_ppl(mqa_model_random, encodings, max_length=max_length)
    pass
    
# group_ppl = group_ppl_calc(group_idxx)

print("base: ", ppl)
#print("base model, random weights: ", ppl_random)
print("base model, LoRA tuned: ", ppl_peft)
#print("MHA -> MQA, transformed weights: ", ppl_mqa)
#print("MHA -> MQA, random weights: ", ppl_mqa_random)
#for i in range(group_ppl):
#    print(f"MHA -> GQA with {len(group_idxx[i])} groups, transformed weights: ", group_ppl[i][0])
#    print(f"MHA -> GQA with {len(group_idxx[i])} groups, random weights: ", group_ppl[i][1], end="\n\n")

