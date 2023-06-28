## TODO (JH) remove models.py in this folder and designate the file in the parent folder.
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import numpy as np
import math

from tqdm import tqdm

from models import load_backbone, Classifier
import src.setup as setup

def cut_input(args, tokens):
    if 'roberta' in args.base_model:
        attention_mask = (tokens != 1).float()
    else:
        attention_mask = (tokens > 0).float()
    max_len = int(torch.max(attention_mask.sum(dim=1)))
    return tokens[:, :max_len], attention_mask[:, :max_len]

def get_features(args, model, loader, mode='test'):
    if mode == 'train':
        model.train()
    else:
        model.eval()

    all_sources = {}
    all_penuls, all_logits, all_probs, all_tokens, all_labels = [], [], [], [], []

    # for i, (tokens, labels, indices) in enumerate(loader):
    for batch in tqdm(loader):
        orig_tokens = batch[0].cuda()
        tokens, attention_mask = cut_input(args, orig_tokens)

        labels = batch[3].cuda()
        #labels = labels.squeeze(1)  # (B)

        with torch.no_grad():
            logits = model(tokens)[0]
            penuls = model.bert(tokens, attention_mask=attention_mask)[1]

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
    ens_sources = []
    for i in list_epoch:
        for j in list_seed:
            # Load ith model
            pre_ckpt_i = os.path.join(args.model_name_or_path, f"seed{j}_epoch{i}")
            model_i = setup.load_model_by_name(args, pre_ckpt_i)

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
