"""
Check the doc at mase-tools/docs/large_language_models/accelerate_fsdp.md
"""

import os
import sys

from accelerate_train import train
from data_module import MyDataModule

# from chop.models.manual.llama_plain.modeling_llama import LlamaForCausalLM
from transformers.models.llama import LlamaTokenizer, LlamaForCausalLM


def main():
    model_name = "Cheng98/llama-160m"
    task = "lm"
    dataset_name = "wikitext2"
    max_token_len = 128
    batch_size = 16
    num_workers = os.cpu_count()
    optimizer = "adamw"
    max_epochs: int = 10
    max_steps: int = -1
    gradient_accumulation_steps: int = 1
    # Reduced for unit test
    # max_epochs: int = 2
    # max_steps: int = -1
    # gradient_accumulation_steps: int = 4
    learning_rate: float = 1.5e-4
    weight_decay: float = 0.0
    lr_scheduler_type: str = "linear"
    num_warmup_steps: int = 0
    save_path: str = "./ckpts"
    load_name: str = None
    load_type: str = ""

    model = LlamaForCausalLM.from_pretrained(model_name)
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
    )


if __name__ == "__main__":
    main()