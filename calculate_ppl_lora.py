import os

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"

from accelerate_peft import train, parse_arguments
from data_module import MyDataModule
from lora_utils import (
    print_trainable_parameters,
    mark_only_lora_as_trainable
)

from configuration_llama_llora import LlamaLoraConfig
from modeling_llama_llora import LlamaForCausalLM
from transformers.models.llama import LlamaTokenizer
import toml
from datasets import load_dataset
import torch
from tqdm import tqdm
import numpy as np


def get_lora_state_from_pretrained(model_to_load, peft_config, num_layers=12):
    """Only works for model of class LlamaForCausalLM from modeling_llama_llora"""
    lora_params_model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_to_load, config=peft_config
    )
    state_dict = lora_params_model.state_dict()
    state = dict()
    for layer_id in range(num_layers):
        for head in ["q", "k", "v", "o"]:
            for lora_mat in ["A", "B"]:
                layer = f'model.layers.{layer_id}.self_attn.{head}_proj.lora_{lora_mat}.eng_alpaca.weight'
                state[layer] = state_dict[layer]

    return state


# Implementation is questionable
def calculate_ppl(input_model, encodings, stride=512, max_length=1024):
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


def evaluate_lm_step(model: torch.nn.Module, batch):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_attention_mask = attention_mask[..., 1:].contiguous()

    # HF's evaluate perplexity: https://huggingface.co/spaces/evaluate-metric/perplexity/blob/main/perplexity.py
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    ppl_step = torch.exp(
        (
            loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask
        ).sum(1)
        / shift_attention_mask.sum(1)
    )
    return ppl_step


def evaluate(model, task, eval_dataloader):
    model.eval()
    step_results = []
    for step, batch in enumerate(eval_dataloader):
        match task:
            case "lm" | "language_modeling":
                ppl_step = evaluate_lm_step(model, batch)
                step_results += ppl_step.tolist()
            case _:
                raise ValueError(f"Unsupported task: {task}")

    match task:
        case "lm" | "language_modeling":
            ppl = np.mean(step_results)
            eval_results = {"eval_ppl": ppl}
        case _:
            raise ValueError(f"Unsupported task: {task}")

    return eval_results


device = 'cuda'
max_length = 1024
model_name = "Cheng98/llama-160m"
# lora_config_path = parse_arguments() --> The following is used to pass a .toml file throught the CLI e.g --lora-config-path machop/configs/by_model/llama_lora/lora_by_type.toml
config_files = [
    "lora_by_type.toml",
]

#peft_config = LlamaLoraConfig.from_pretrained(
#    pretrained_model_name_or_path=model_name, lora_config=lora_config_path
#)
#peft_config.task_type = TaskType.CAUSAL_LM
index = int(input("Index of the lora param file: "))
if index == -1:
    index = len(os.listdir("ckpts-llama-lora-plain/")) - 1
model_to_load = f"ckpts-llama-lora-plain/{index}"
for config_file in config_files:
    # load toml config file
    with open(config_file, "r") as f:
        lora_config_path = toml.load(f)
    print(f"LoRA PEFT with {config_file} config file successfully loaded!")

print(f"Loaded lora PARAMS from {model_to_load}")
peft_config = LlamaLoraConfig.from_pretrained(
    pretrained_model_name_or_path=model_to_load, lora_config=lora_config_path
)
peft_model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name, config=peft_config
)

peft_model.load_state_dict(get_lora_state_from_pretrained(model_to_load, peft_config), strict=False)
peft_mdoel = peft_model.to(device)

# Prepare & encode data
tokenizer = LlamaTokenizer.from_pretrained(model_name)
test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = test.map(lambda x: tokenizer(x["text"], return_tensors="pt"))
encodings = merge_list(encodings["input_ids"])
encodings = merge_list(encodings)
encodings = torch.tensor(encodings).reshape(1, -1).to(device)

data_module = MyDataModule(
    model_name=None,
    dataset_name="wikitext2",
    batch_size=1,
    workers=1,
    tokenizer=tokenizer,
    max_token_len=128,
).to(device)
data_module.prepare_data()
data_module.setup()
eval_dataloader = data_module.val_dataloader()
eval_results = evaluate(peft_model, 'lm', eval_dataloader)

print("mase's eval_ppl: ", eval_results)
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

