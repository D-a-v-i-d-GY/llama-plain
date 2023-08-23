import torch
import os

from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from datasets import load_dataset
import transformers
from peft import get_peft_model, LoraConfig

def main():

    # Model Parameters
    model_name = "Cheng98/llama-160m"
    batch_size = 4
    max_epochs: int = 3
    max_steps: int = -1
    r = 8
    lora_alpha = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_warmup_steps: int = 0
    
    # Load the model and tokenizer
    model = LlamaForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",
    )

    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    # Freeze non-LoRA parameters 
    #for param in model.parameters():
    #    param.requires_grad = False
    #    if param.ndim == 1:
    #        param.data = param.data.to(torch.float32)

    # Peft config for LoRA
    peft_config = LoraConfig(
        r=r, 
        lora_alpha=lora_alpha, 
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, 
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Dataset
    train_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    train_data_enc = train_data.map(lambda x: tokenizer(x["text"]), batched=True)

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data_enc,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=num_warmup_steps,
            num_train_epochs=max_epochs,
            max_steps=max_steps,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=3000,
            output_dir="my-lora-train-ckpts",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )


    model.config.use_cache=False
    trainer.train()

    index = len(os.listdir("lora_models/"))
    model_id = f"lora_models/plain-lora-{index}"

    model_to_save = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    print(model_to_save)
    model_to_save.save_pretrained(model_id)

    #model.save_pretrained(model_id)

if __name__ == "__main__":
    main()
