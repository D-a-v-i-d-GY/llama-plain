import torch
import os

from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from datasets import load_dataset
import transformers
from peft import get_peft_model, LoraConfig

def main():

    model_name = "Cheng98/llama-160m"
    batch_size = 4
    max_epochs: int = 1
    max_steps: int = -1
    r = 4
    lora_alpha = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    num_warmup_steps: int = 0
    
    model = LlamaForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="auto",
    )

    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

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

    train_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    train_data_enc = train_data.map(lambda x: tokenizer(x["text"]))

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
            logging_steps=None,
            load_best_model_at_end=True,
            output_dir="my-lora-train-ckpts",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )


    model.config.use_cache=False
    trainer.train()

    index = len(os.listdir("lora_models/"))
    model_id = f"lora_models/plain-lora-{index}"
    model.save_pretrained(model_id)

if __name__ == "__main__":
    main()
