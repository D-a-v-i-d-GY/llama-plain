import torch
from typing import Optional

# Use a pipeline as a high-level helper
from transformers import pipeline
# Load model directly
from transformers.models.llama import LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

gpu = torch.device("cuda:0")
tokenizer = LlamaTokenizer.from_pretrained("Cheng98/llama-160m")
model = AutoModelForCausalLM.from_pretrained("Cheng98/llama-160m")
pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    framework="pt", 
    device=gpu,
    )
print(pipe("", return_full_text=True, max_new_tokens=50)[0]["generated_text"])