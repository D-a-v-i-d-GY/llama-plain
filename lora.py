import torch
import os

# Load model directly
from transformers.models.llama import LlamaTokenizerFast
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import transformers
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from data_module import MyDataModule
from accelerate_train import train

def main():

    model_name = "Cheng98/llama-160m"
    #task = "lm"
    dataset_name = "wikitext2"
    max_token_len = 128
    batch_size = 1
    num_workers = 0
    #optimizer = "adamw"
    #max_epochs: int = 1
    max_steps: int = 1
    gradient_accumulation_steps: int = 1
    # Reduced for unit test
    # max_epochs: int = 2
    # max_steps: int = -1
    # gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    #lr_scheduler_type: str = "linear"
    num_warmup_steps: int = 0
    #save_path: str = "./ckpts"
    #load_name: str = None
    #load_type: str = ""
    #gpu = torch.device("cuda:0")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = LlamaTokenizerFast.from_pretrained("decapoda-research/llama-7b-hf")
    tokenizer.add_special_tokens({"pad_token":"<PAD>"})

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],
        inference_mode=False, 
        r=8, 
        lora_alpha=16, 
        lora_dropout=0.1, 
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_data = load_dataset("wikipedia", "20220301.en")

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=num_warmup_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            weight_decay=weight_decay,
            dataloader_pin_memory=True,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )


    model.config.use_cache=False
    trainer.train()

if __name__ == "__main__":
    main()
