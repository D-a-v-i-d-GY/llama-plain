from transformers import pipeline
from transformers.models.llama import LlamaTokenizer, LlamaForCausalLM
import modeling_llama_mqa

model_name = "Cheng98/llama-160m"

model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name)
my_mqa_model = modeling_llama_mqa.LlamaForCausalLM(model.config)

generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=25)
print(generator("Hello,")[0]["generated_text"])

generator = pipeline(task="text-generation", model=my_mqa_model, tokenizer=tokenizer, max_new_tokens=25)
print(generator("Hello,")[0]["generated_text"])

#my_mqa_model.load_state_dict(model.state_dict())
#generator = pipeline(task="text-generation", model=my_mqa_model, tokenizer=tokenizer, max_new_tokens=15)
#print(generator("Hello, W")[0]["generated_text"])