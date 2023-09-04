import optuna
from optuna_dashboard import run_server
import torch
from architecture_transform_util import mha2gqa
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.tokenization_llama import LlamaTokenizer
import os
from tqdm import tqdm
import modeling_llama_gqa
from data_module import MyDataModule
import numpy as np
import math
from optuna.trial import TrialState


def evaluate_lm_step(model: torch.nn.Module, batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_attention_mask = attention_mask[..., 1:].contiguous()

    # HF's evaluate perplexity: https://huggingface.co/spaces/evaluate-metric/perplexity/blob/main/perplexity.py
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    ppl_step = torch.exp(
        (
            loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask
        ).sum(1)
        / shift_attention_mask.sum(1)
    )
    return ppl_step


def evaluate(model, task, eval_dataloader, device, step_stop=-1):
    model.eval()
    step_results = []
    for step, batch in enumerate(eval_dataloader):
        if  step_stop != -1 and step == step_stop: 
            break
        match task:
            case "lm" | "language_modeling":
                ppl_step = evaluate_lm_step(model, batch, device)
                step_results += ppl_step.tolist()
            case _:
                raise ValueError(f"Unsupported task: {task}")

    match task:
        case "lm" | "language_modeling":
            ppl = np.mean(step_results)
            eval_results = {"eval_ppl": ppl}
        case _:
            raise ValueError(f"Unsupported task: {task}")

    return eval_results


def n_uniform_groups(num_groups, num_heads, num_layers, depth=-1, reverse=False):
    group_size = num_heads // num_groups
    #if reverse: print(f"Reversed uniform grouping with {num_groups} groups, group size = {group_size}, and grouping layer depth = {depth}")
    #else: print(f"Uniform grouping with {num_groups} groups, group size = {group_size}, and grouping layer depth = {depth}")
    mha = [[[i] for i in range(num_heads)]]
    reverse = 1 if reverse else 0

    if group_size * num_groups != num_heads:
        raise ValueError(f"number of heads: {num_heads} must be divisible by number of groups: {num_groups}")
    
    if depth == -1:
        return [[list(range(group_size * i, group_size * (i + 1))) for i in range(num_groups)]] * num_layers
    elif depth <= num_layers and depth >= 0:
        return (mha * ((num_layers - depth) * reverse)
                + [[list(range(group_size * i, group_size * (i + 1))) for i in range(num_groups)]] * depth 
                + mha * ((num_layers - depth) * (1 - reverse)))


def ids2group_idx(ids):
    depth = len(ids) # ids: list of size depth * num_heads
    num_heads = 12
    group_idx = [[] for q in range(depth)]
    for i in range(depth):
        layer_groups = [[] for j in range(num_heads)]
        for j in range(num_heads):
            layer_groups[ids[i][j]] += [j]
        layer_groups = [group for group in layer_groups if group]
        group_idx[i] = layer_groups
    
    return group_idx


def fill_group_idx(group_idx, num_heads, num_layers, reverse=False):
    mha = [[[i] for i in range(num_heads)]]
    reverse = 1 if reverse else 0
    depth = len(group_idx)

    return (mha * ((num_layers - depth) * reverse)
            + group_idx
            + mha * ((num_layers - depth) * (1 - reverse)))


def objective(trial):

    # Select the architecture of the gqa model based on optuna suggestion
#    grpsz_index = trial.suggest_int("group_size_index", 0, 5)
#    num_groups = possible_num_of_groups[grpsz_index]
#    num_groups = trial.suggest_categorical("number of groups", [1, 2, 3, 4, 6])
#    depth = trial.suggest_int("grouping depth", 1, 12)
#    rev = trial.suggest_categorical("Grouping from the back", [True, False])
    layer_ids = [[trial.suggest_int(f"group of {i}-th head in {j}-th layer", 0, 11) for i in range(12)] for j in range(12)]
    group_idx = ids2group_idx(layer_ids)
    num_groups = sum([len(layer_groups) for layer_groups in group_idx])
#    group_idx = fill_group_idx(group_idx, num_heads=12, num_layers=12)#, reverse=rev)
#    group_idx = n_uniform_groups(num_groups, 12, 12, depth=depth, reverse=rev)

    # Print group_idx
    for layer_groups in group_idx:
        print(layer_groups)

    # GQA model init
    model.config.groups_idx = group_idx
    gqa_model = modeling_llama_gqa.LlamaForCausalLM(model.config).to(device)
    state = model.state_dict()
    gqa_model.load_state_dict(mha2gqa(state, group_idx, num_heads=12, transpose_layer=True))
    
    # Calculate the ppl, don't go thruo the whole dataset
    eval_results = evaluate(gqa_model, 'lm', eval_dataloader, device, step_stop=num_of_evals)
    print(eval_results["eval_ppl"])
    # Evaluate the objective function based on ppl and grouping complexity
    return eval_results["eval_ppl"] * num_groups ** 0.8 


model_name = "Cheng98/llama-160m"
device = "cuda"
possible_num_of_groups = [1, 2, 3, 4, 6]
grouping_depth = 1
model = LlamaForCausalLM.from_pretrained(model_name).to(device)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# Prepare data for evaluation
data_module = MyDataModule(
    model_name=None,
    dataset_name="wikitext2",
    batch_size=1,
    workers=0,
    tokenizer=tokenizer,
    max_token_len=128,
)
data_module.prepare_data()
data_module.setup()
eval_dataloader = data_module.val_dataloader()
num_of_evals = len(eval_dataloader.dataset) // 10

# Create the optimisation study
study = optuna.create_study(
    sampler=optuna.samplers.RandomSampler(),
    storage="sqlite:///test.db",
    study_name="test-study",
    load_if_exists=True,
    direction="minimize",
)

study.optimize(objective, n_trials=50, show_progress_bar=True)

optuna.visualization.plot_param_importances(study)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
