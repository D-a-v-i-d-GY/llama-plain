# Research project (David Gyulamiryan, email: dg615@cam.ac.uk)

### Timeline: I worked on this project from 03/07/23 to 08/09/23

# Aims and methods
The goal of the project was to reduce inference memory consumption of pre-trained transformers (here small version of llama was used, with 160m params) by changing the architecture of attention layers post-training. The proposed method: asymmetric grouping of K and V heads within each attention layer (based on their performance similarity to preserve as much info as possible), then fine-tuning using LoRA. The method is an extension of the method from arXiv:2305.13245v1 (GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints) paper.

# Code
A lot of the code was borrowed from mase-tools, and then further modified for the project. The code was written with experimental purpose 
