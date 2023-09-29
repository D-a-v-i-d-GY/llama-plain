# Research project (David Gyulamiryan, email: dg615@cam.ac.uk)

### Timeline: I worked on this project from 03/07/23 to 08/09/23

# Aims and methods
The goal of the project was to reduce inference memory consumption of pre-trained transformers (here small version of llama was used, with 160m params) by changing the architecture of attention layers post-training. The proposed method: asymmetric grouping of K and V heads within each attention layer (based on their performance similarity to preserve as much info as possible), then fine-tuning using LoRA. The method is an extension of the method from arXiv:2305.13245v1 (GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints) paper.

# Code
A lot of the code was borrowed from mase-tools (Machine-Learning Accelerator System Exploration Tools, led by one of my supervisors' research team), and then further modified for this project. There are also some Colab notebooks that I used in the beginning of the project, I will upload these later.

### Training
- `accelerate_fsdp_llama.py` used for training the base 160m llama model
- `accelerate_fsdp_llama_lora.py` fine-tuning the base 160m llama model using LoRA
- `accelerate_fsdp_llama_lora_gqa.py` fine-tuning the asymmetrically grouped 160m llama model using LoRA
- `lora.py` early code for LoRA fine-tuning

### Model description/architecture change
- `architecture_transform_util.py` change the architecture of a pre-trained model, from both plain and LoRA fine-tuned chechpoint, to asymmetric-GQA
- `modelling_llama.py` the entire model description of plain llama model 
- `modelling_llama_gqa.py` the entire model description of asymmetric GQA llama model
- `modelling_llama_gqa_lora.py` the entire model description of asymmetric GQA llama model modified for LoRA fine-tuning
- `modelling_llama_llora.py` the entire model description of plain llama model for LoRA fine-tuning
- `modelling_llama_mqa.py` the entire model description of MQA model, early implementation of grouping, not used after the asymmetric GQA model was implemented

### Testing and evaluation
- `inference_test.py`, `language_modeling.py` and `test.py` general testing, first two files not updated for a long time
- `calculate_ppl.py` and `calculate_ppl_lora.py` are used to calculate the perplexity of the model using two different methods (one file for base models, the other for LoRA fine-tuned), though first method is preferred (is printed first), at some point both were changed to work only with LoRA, not sure how this happened :)

The files listed above are the ones you would usually execute. All the other files are used within these, or not used at all.

# State of the project upon finishing the internship
After implementing the flexible/asymmetric grouping of attention heads, next step was identifying similar heads to group. Initially random grouping was decided to be used. `gqa_optuna.py` was the latest piece of code I was working on. It would use optuna library to find overall grouping that would minimize an objective function `ppl * tot_num_of_groups ** 2`. The optimal groupings would be then studied to find if there are certain heads that if grouped produce relatively better results. Best result: after finding an optimal groupings with about 100 total groups (i.e. 2/3 of the original size), the model was fine-tuned with LoRA for 5 epochs on Wikitext2, the ppl value of this model was 26.2, while the original model had ppl value of 20.6 after similar fine-tuning.

# Next steps, issues, etc.
...
