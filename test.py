import torch
from typing import Optional

# Use a pipeline as a high-level helper
from transformers import pipeline
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

pipe = pipeline("text-generation", model="Cheng98/llama-160m")

tokenizer = AutoTokenizer.from_pretrained("Cheng98/llama-160m")
model = AutoModelForCausalLM.from_pretrained("Cheng98/llama-160m")
