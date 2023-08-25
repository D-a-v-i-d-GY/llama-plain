from datasets import load_dataset
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.tokenization_llama import LlamaTokenizer
import torch
import os
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, PeftModel
import modeling_llama_mqa
import modeling_llama_gqa
from architecture_transform_util import mha2mqa, mha2gqa
import toml
from lora_utils import print_trainable_parameters


def calculate_ppl(input_model, encodings, stride=512, max_length=2048):
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
            outputs = input_model(input_ids, labels=target_ids)

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


model_name = "Cheng98/llama-160m"
# model_name = "lmsys/vicuna-7b-v1.3"
# lora_config_path = parse_arguments() --> The following is used to pass a .toml file throught the CLI e.g --lora-config-path machop/configs/by_model/llama_lora/lora_by_type.toml
config_files = [
    "lora_by_type.toml",
]
task = "lm"
dataset_name = "wikitext2"
max_token_len = 128
batch_size = 16
num_workers = os.cpu_count()
optimizer = "adamw"
max_epochs: int = 2
max_steps: int = -1
gradient_accumulation_steps: int = 1
# Reduced for unit test
# max_epochs: int = 2
# max_steps: int = -1
# gradient_accumulation_steps: int = 4
learning_rate: float = 5e-5
weight_decay: float = 0.01
lr_scheduler_type: str = "linear"
num_warmup_steps: int = 0
save_path: str = "./ckpts-lora-plain"
load_name: str = None
load_type: str = ""

for config_file in config_files:
    # load toml config file
    with open(config_file, "r") as f:
        lora_config_path = toml.load(f)
    print(f"LoRA PEFT with {config_file} config file successfully loaded!")

#peft_config = LlamaLoraConfig.from_pretrained(
#    pretrained_model_name_or_path=model_name, lora_config=lora_config_path
#)
#peft_config.task_type = TaskType.CAUSAL_LM

index = len(os.listdir("ckpts-lora-plain/"))
lora_model_id = f"ckpts-lora-plain/{index - 1}"
print(f"Loading {lora_model_id}")
lora_config = LoraConfig.from_pretrained(lora_model_id)

model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name, # config=peft_config
)
peft_model = get_peft_model(model, lora_config)

print_trainable_parameters(model)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

device = 'cuda'
max_length = 1024

# Prepare & encode data
tokenizer = LlamaTokenizer.from_pretrained(model_name)
test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = test.map(lambda x: tokenizer(x["text"], return_tensors="pt"))
encodings = merge_list(encodings["input_ids"])
encodings = merge_list(encodings)
encodings = torch.tensor(encodings).reshape(1, -1).to(device)

# LoRA model (latest)
#model = LlamaForCausalLM.from_pretrained(model_name).to(device)
#index = len(os.listdir("lora_models/"))
#lora_model_id = f"lora_models/plain-lora-{index - 1}"
#print(f"Loading {lora_model_id}")
#lora_config = LoraConfig.from_pretrained(lora_model_id)
#peft_model = get_peft_model(model, lora_config).to(device)

#lora_B_layer = peft_model.state_dict()['base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight']
#print(torch.where(lora_B_layer != torch.zeros_like(lora_B_layer)))
#print(lora_config)

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
#    model.eval()
    peft_model.eval()
#    mqa_model.eval()
#    mqa_model_random.eval()

#    ppl = calculate_ppl(model, encodings, max_length=max_length)
    ppl_peft = calculate_ppl(peft_model, encodings, max_length=max_length)
 #   ppl_random = calculate_ppl(model_random, encodings, max_length=max_length)
#    ppl_mqa = calculate_ppl(mqa_model, encodings, max_length=max_length)
#    ppl_mqa_random = calculate_ppl(mqa_model_random, encodings, max_length=max_length)
    pass
    
# group_ppl = group_ppl_calc(group_idxx)

#print("base: ", ppl)
print("base model, LoRA tuned: ", ppl_peft)
#print("base model, random weights: ", ppl_random)
#print("MHA -> MQA, transformed weights: ", ppl_mqa)
#print("MHA -> MQA, random weights: ", ppl_mqa_random)
#for i in range(group_ppl):
#    print(f"MHA -> GQA with {len(group_idxx[i])} groups, transformed weights: ", group_ppl[i][0])
#    print(f"MHA -> GQA with {len(group_idxx[i])} groups, random weights: ", group_ppl[i][1], end="\n\n")

