import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import numpy as np
import math

from src.models import load_backbone, Classifier
from src.utils import cut_input

def get_features(args, model, loader, mode='test'):
    if mode == 'train':
        model.train()
    else:
        model.eval()

    all_sources = {}
    all_penuls, all_logits, all_probs, all_tokens, all_labels = [], [], [], [], []

    for i, (tokens, labels, indices) in enumerate(loader):
        orig_tokens = tokens.cuda()
        tokens, _ = cut_input(args, orig_tokens)

        labels = labels.cuda()
        labels = labels.squeeze(1)  # (B)

        with torch.no_grad():
            logits, penuls = model(tokens, inputs_embed=None, get_penul=True)

        all_penuls.append(penuls.cpu())
        all_logits.append(logits.cpu())
        all_probs.append(logits.softmax(dim=1).cpu())
        all_labels.append(labels.cpu())
        all_tokens.append(orig_tokens.cpu())

    all_penuls = torch.cat(all_penuls, dim=0)
    all_logits = torch.cat(all_logits, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_tokens = torch.cat(all_tokens, dim=0)

    all_sources['penuls'] = all_penuls
    all_sources['logits'] = all_logits
    all_sources['probs'] = all_probs
    all_sources['labels'] = all_labels
    all_sources['tokens'] = all_tokens

    return all_sources

def merge_multiple_models(args, loader, backbone, list_epoch, list_seed):
    n_epochs = len(list_epoch)
    n_seeds = len(list_seed)

    ens_sources = []
    for i in range(n_epochs):
        for j in range(n_seeds):
            # Load ith model
            pre_ckpt_i = f"./logs/{args.dataset}_R{args.data_ratio}_{args.backbone}_{args.train_type}_S{list_seed[j]}/"
            pre_ckpt_i += f"epoch{list_epoch[i]}.model"

            model_i = Classifier(args.backbone, backbone, args.n_class, args.train_type).cuda()
            ckpt_dict = torch.load(args.pre_ckpt)
            model_i.load_state_dict(ckpt_dict)
            
            # Extracting its features
            sources_i = get_features(args, model_i, loader)
            ens_sources.append(sources_i)

    keys = list(ens_sources[0].keys())  # exception: labels
    n_models = len(ens_sources)
    ens_sources_m = {}
    for key in keys[:-2]:
        temp = []
        for t in range(n_models):
            temp.append(ens_sources[t][key].unsqueeze(0))
        temp = torch.cat(temp, dim=0)

        ens_sources_m[key] = temp

    # Adding tokens & label to dictionary
    ens_sources_m[keys[-2]] = ens_sources[0][keys[-2]]
    ens_sources_m[keys[-1]] = ens_sources[0][keys[-1]]

    return ens_sources_m
