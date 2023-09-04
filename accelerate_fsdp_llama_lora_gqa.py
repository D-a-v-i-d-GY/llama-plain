"""
Check the doc at mase-tools/docs/large_language_models/accelerate_fsdp.md
"""

import os

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"

from accelerate_peft import train, parse_arguments
from data_module import MyDataModule
from lora_utils import (
    print_trainable_parameters,
    mark_only_lora_as_trainable
)

from configuration_llama_llora import LlamaLoraConfig
import modeling_llama_llora
import modeling_llama_gqa_lora
from transformers.models.llama import LlamaTokenizer
import toml
from architecture_transform_util import mha2gqa_lora
import torch
import numpy as np


def evaluate_lm_step(model: torch.nn.Module, batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
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


def evaluate(model, task, eval_dataloader, device):
    model.eval()
    step_results = []
    for step, batch in enumerate(eval_dataloader):
        match task:
            case "lm" | "language_modeling":
                ppl_step = evaluate_lm_step(model, batch, device)
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


def main():
    group_idx = [ 
                [[0, 5], [2], [1, 7], [6], [8], [4, 10, 11], [9], [3]],
                [[4], [3], [9], [2], [0], [10, 11], [5], [1, 6], [7, 8]],
                [[5], [1, 6], [3, 4], [8, 10], [2], [0, 7], [9, 11]],
                [[6], [0], [1, 2, 7], [10], [5, 9], [4], [8, 11], [3]],
                [[8], [0, 10, 11], [2], [4, 6], [1], [3, 7], [5, 9]],
                [[11], [10], [5], [3, 4], [6, 8], [0], [2, 7], [1, 9]],
                [[1, 6, 10], [5, 7], [4], [0, 9, 11], [8], [3], [2]],
                [[0, 6], [2, 3, 5, 8], [4], [1], [11], [9], [10], [7]],
                [[8], [0, 3, 4, 5], [6, 7], [9], [2], [10], [11], [1]],
                [[0], [11], [8], [1, 10], [2, 5], [3], [9], [6, 7], [4]],
                [[0], [2], [5, 8], [6], [3], [1], [7], [4], [9], [10, 11]],
                [[10], [0], [11], [6, 7], [2, 3, 5], [1, 8], [4], [9]]
                ]
    model_name = "Cheng98/llama-160m"
    # model_name = "lmsys/vicuna-7b-v1.3"
    # lora_config_path = parse_arguments() --> The following is used to pass a .toml file throught the CLI e.g --lora-config-path machop/configs/by_model/llama_lora/lora_by_type.toml
    config_files = [
        "lora_by_type_gqa.toml",
    ]
    task = "lm"
    dataset_name = "wikitext2"
    max_token_len = 128
    batch_size = 4
    num_workers = os.cpu_count()
    optimizer = "adamw"
    max_epochs: int = 5
    max_steps: int = -1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    lr_scheduler_type: str = "linear"
    num_warmup_steps: int = 0
    save_path: str = "./ckpts-llama-lora-gqa"
    load_name: str = None
    load_type: str = ""

    for config_file in config_files:
        # load toml config file
        with open(config_file, "r") as f:
            lora_config_path = toml.load(f)
        print(f"LoRA PEFT with {config_file} config file successfully loaded!")

    print("\nGroupings:")
    for layer_groups in group_idx:
        print(layer_groups)
    print("Number of groups: ", sum([len(layer_groups) for layer_groups in group_idx]))

    peft_config = LlamaLoraConfig.from_pretrained(
        pretrained_model_name_or_path=model_name, lora_config=lora_config_path
    )
    model = modeling_llama_llora.LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name, config=peft_config
    )
    model.config.groups_idx = group_idx
    gqa_model = modeling_llama_gqa_lora.LlamaForCausalLM(model.config)
    
    state = model.state_dict()
    gqa_model.load_state_dict(mha2gqa_lora(state, group_idx, num_heads=12, transpose_layer=True))

    gqa_model = mark_only_lora_as_trainable(gqa_model)
    print_trainable_parameters(gqa_model)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    data_module = MyDataModule(
        model_name=None,
        dataset_name=dataset_name,
        batch_size=batch_size,
        workers=num_workers,
        tokenizer=tokenizer,
        max_token_len=max_token_len,
    )

    device = "cuda"
    data_module.prepare_data()
    data_module.setup()
    eval_dataloader = data_module.val_dataloader()
    eval_results = evaluate(gqa_model, 'lm', eval_dataloader, device)

    print("eval_ppl before fine-tuning: ", eval_results)

    train(
        model=gqa_model,
        task=task,
        data_module=data_module,
        optimizer=optimizer,
        max_epochs=max_epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr_scheduler_type=lr_scheduler_type,
        num_warmup_steps=num_warmup_steps,
        save_path=save_path,
        load_name=load_name,
        load_type=load_type,
#        evaluate_before_training=evaluate_before_training,
    )


if __name__ == "__main__":
    main()
