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
from modeling_llama_llora import LlamaForCausalLM
from transformers.models.llama import LlamaTokenizer
import toml


def main():
    model_name = "Cheng98/llama-160m"
    # model_name = "lmsys/vicuna-7b-v1.3"
    # lora_config_path = parse_arguments() --> The following is used to pass a .toml file throught the CLI e.g --lora-config-path machop/configs/by_model/llama_lora/lora_by_type.toml
    config_files = [
        "lora_by_type.toml",
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
    save_path: str = "./ckpts-llama-lora-plain"
    load_name: str = None
    load_type: str = ""

    for config_file in config_files:
        # load toml config file
        with open(config_file, "r") as f:
            lora_config_path = toml.load(f)
        print(f"LoRA PEFT with {config_file} config file successfully loaded!")

    peft_config = LlamaLoraConfig.from_pretrained(
        pretrained_model_name_or_path=model_name, lora_config=lora_config_path
    )
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name, config=peft_config
    )
    model = mark_only_lora_as_trainable(model)
    print_trainable_parameters(model)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    data_module = MyDataModule(
        model_name=None,
        dataset_name=dataset_name,
        batch_size=batch_size,
        workers=num_workers,
        tokenizer=tokenizer,
        max_token_len=max_token_len,
    )

    train(
        model=model,
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
