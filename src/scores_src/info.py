import os
import easydict
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import scipy
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader, TensorDataset

from src.models import load_backbone, Classifier
from src.data import get_base_dataset
from src.utils import Logger, set_seed, set_model_path, save_model
from src.training.common import cut_input

from src.scores_src import get_features, merge_multiple_models, surprisal_embed, surprisal_embed_wino
from src.scores_src import avg_conf_variab, avg_forgetting, avg_aum
from src.scores_src import get_density_score, get_sentence_embedding
from src.scores_src import confidence, entropy, badge_grads_norm, badge_grads
from src.scores_src import mc_dropout_models, el2n_score, ens_max_ent, ens_bald, ens_varR
from src.scores_src import gaussian_kernel, dpp_greedy

def aggregate(args, features, features_t, alps, alps_t, ens_features_epochs, ens_features_models, mc_ens_features,
              knn_density_s_np):
    all_measurements = []
    name = []

    # Training dynamics
    avg_conf, variab = avg_conf_variab(ens_features_epochs)
    forget_number = avg_forgetting(ens_features_epochs)
    aum = avg_aum(ens_features_epochs)

    all_measurements.append(avg_conf.unsqueeze(0))
    all_measurements.append(variab.unsqueeze(0))
    all_measurements.append(forget_number.unsqueeze(0))
    all_measurements.append(aum.unsqueeze(0))

    name.append('avg_conf')
    name.append('variab')
    name.append('forget_number')
    name.append('aum')

    # Model Ensemble
    ens_el2n = el2n_score(ens_features_models)
    ens_ent = ens_max_ent(ens_features_models)
    ens_BALD = ens_bald(ens_features_models)
    ens_VAR = ens_varR(ens_features_models)
    ens_avg_conf, ens_variab = avg_conf_variab(ens_features_models)

    all_measurements.append(ens_el2n.unsqueeze(0))
    all_measurements.append(ens_ent.unsqueeze(0))
    all_measurements.append(ens_BALD.unsqueeze(0))
    all_measurements.append(ens_VAR.unsqueeze(0))
    all_measurements.append(ens_avg_conf.unsqueeze(0))
    all_measurements.append(ens_variab.unsqueeze(0))

    name.append('ens_el2n')
    name.append('ens_ent')
    name.append('ens_BALD')
    name.append('ens_VAR')
    name.append('ens_avg_conf')
    name.append('ens_variab')

    # MC Ensemble
    mc_ens_el2n = el2n_score(mc_ens_features)
    mc_ens_ent = ens_max_ent(mc_ens_features)
    mc_ens_BALD = ens_bald(mc_ens_features)
    mc_ens_varR = ens_varR(mc_ens_features)
    mc_ens_avg_conf, mc_ens_variab = avg_conf_variab(mc_ens_features)

    all_measurements.append(mc_ens_el2n.unsqueeze(0))
    all_measurements.append(mc_ens_ent.unsqueeze(0))
    all_measurements.append(mc_ens_BALD.unsqueeze(0))
    all_measurements.append(mc_ens_varR.unsqueeze(0))
    all_measurements.append(mc_ens_avg_conf.unsqueeze(0))
    all_measurements.append(mc_ens_variab.unsqueeze(0))

    name.append('mc_ens_el2n')
    name.append('mc_ens_ent')
    name.append('mc_ens_BALD')
    name.append('mc_ens_VAR')
    name.append('mc_ens_avg_conf')
    name.append('mc_ens_variab')

    # Single Model
    conf = -1 * confidence(features)
    ent = entropy(features)
    badge_embed = badge_grads(features)
    badge_norm = badge_embed.norm(dim=-1)

    args.density_measure = 'nn_relative_dist'
    knn_density_rel = get_density_score(args, features['penuls'].numpy(), features['labels'].numpy(),
                                        features_t['penuls'].numpy(), features_t['labels'].numpy())
    args.density_measure = 'nn_dist'
    knn_density = get_density_score(args, features['penuls'].numpy(), features['labels'].numpy(),
                                    features_t['penuls'].numpy(), features_t['labels'].numpy())

    all_measurements.append(torch.Tensor(knn_density).unsqueeze(0))
    all_measurements.append(torch.Tensor(knn_density_rel).unsqueeze(0))
    all_measurements.append(conf.unsqueeze(0))
    all_measurements.append(ent.unsqueeze(0))
    all_measurements.append(badge_norm.unsqueeze(0))

    name.append('knn_density')  # 16
    name.append('knn_density_rel')  # 17
    name.append('conf')  # 18
    name.append('ent')  # 19
    name.append('badge_norm')  # 20

    # Language
    surprisals, likelihood, _ = alps
    surprisals_t, _, _ = alps_t

    sent_density = torch.Tensor(knn_density_s_np).unsqueeze(0)
    likelihoods = -1 * likelihood.unsqueeze(0)

    all_measurements.append(sent_density)  # 21
    all_measurements.append(likelihoods) # 22

    name.append('sent_density')
    name.append('likelihood')

    return all_measurements, name


def get_infoverse(args, label_dataset, pool_dataset, n_epochs, seeds_list, n_class, active=False):
    '''
    label_dataset; data of existing labeled samples
    pool_dataset; data of query pool of unlabeled samples
    '''
    print("==================== (0/5) Start InfoVerse Contruction ===============")

    label_loader = DataLoader(label_dataset, shuffle=False, drop_last=False, batch_size=args.batch_size, num_workers=4)
    pool_loader = DataLoader(pool_dataset, shuffle=False, drop_last=False, batch_size=args.batch_size, num_workers=4)

    backbone, tokenizer = load_backbone(args.backbone)
    model = Classifier(args.backbone, backbone, n_class, args.train_type).cuda()
    ckpt_dict = torch.load(args.pre_ckpt)
    model.load_state_dict(ckpt_dict)

    features_l = get_features(args, model, label_loader)
    features_p = get_features(args, model, pool_loader)
    print("==================== (1/5) Inference for Static Measures is Done ===============")

    labels_l = features_l['labels'] # True label
    pseudo_labels_p = features_p['probs'].max(dim=1)[1]  # Pseudo label

    list_epochs = list(np.arange(n_epochs) + 1)
    list_seeds = [seeds_list[0]]
    ens_features_epochs = merge_multiple_models(args, pool_loader, backbone, list_epochs, list_seeds)
    print("==================== (2/5) Inference for Training Dynamics is Done ===============")

    list_epochs = [n_epochs]
    list_seeds = seeds_list
    ens_features_models = merge_multiple_models(args, pool_loader, backbone, list_epochs, list_seeds)
    print("==================== (3/5) Inference for Ensemble Model Uncertainty is Done ===============")

    # Caution. Re-initialization is necessary
    model = Classifier(args.backbone, backbone, n_class, args.train_type).cuda()
    ckpt_dict = torch.load(args.pre_ckpt)
    model.load_state_dict(ckpt_dict)

    mc_ens_features = mc_dropout_models(args, model, pool_loader, n_ensemble=len(seeds_list))
    print("==================== (4/5) Inference for MC Model Uncertainty is Done ===============")

    if active:
        # Label converting
        features_p['labels'] = features_p['logits'].max(dim=1)[1]
        ens_features_epochs['labels'] = features_p['logits'].max(dim=1)[1]
        ens_features_models['labels'] = features_p['logits'].max(dim=1)[1]
        mc_ens_features['labels'] = features_p['logits'].max(dim=1)[1]

    model = Classifier(args.backbone, backbone, n_class, args.train_type).cuda()
    if args.dataset == 'wino':
        alps_p = surprisal_embed_wino(args, model, pool_loader)
        alps_l = surprisal_embed_wino(args, model, label_loader)
    else:
        alps_p = surprisal_embed(args, model, pool_loader)
        alps_l = surprisal_embed(args, model, label_loader)

    backbone_sent, tokenizer_sent = load_backbone('sentence_bert')
    model = Classifier('sentence_bert', backbone_sent, n_class, args.train_type).cuda()

    label_tokens, label_labels, label_indices  = label_dataset[:]
    pool_tokens, pool_labels, pool_indices = pool_dataset[:]
    label_tokens_sent = change_tokenization(label_tokens, tokenizer, tokenizer_sent)
    pool_tokens_sent = change_tokenization(pool_tokens, tokenizer, tokenizer_sent)

    sent_label_dataset = TensorDataset(label_tokens_sent, label_labels, label_indices)
    sent_pool_dataset = TensorDataset(pool_tokens_sent, pseudo_labels_p, pool_indices)

    sent_label_loader = DataLoader(sent_label_dataset, shuffle=False, drop_last=False, batch_size=args.batch_size, num_workers=4)
    sent_pool_loader = DataLoader(sent_pool_dataset, shuffle=False, drop_last=False, batch_size=args.batch_size, num_workers=4)

    sent_bert_embed_l = get_sentence_embedding(args, model, sent_label_loader, aug_src=None, head='SC')[0]
    sent_bert_embed_p = get_sentence_embedding(args, model, sent_pool_loader, aug_src=None, head='SC')[0]

    print("==================== (5/5) Inference for Pre-trained Knowledge is Done ===============")

    args.density_measure = 'nn_dist'
    knn_density = get_density_score(args, sent_bert_embed_p.numpy(), pseudo_labels_p.numpy(), sent_bert_embed_l.numpy(), labels_l.numpy())

    res, name = aggregate(args, features_p, features_l, alps_p, alps_l, ens_features_epochs, ens_features_models,
                          mc_ens_features, knn_density)

    return torch.cat(res, dim=0).t()


def change_tokenization(tokens, tokenizer_bef, tokenizer_aft):
    n_samples = len(tokens)

    tokens_aft = []
    for i in range(n_samples):
        sent_i = tokenizer_bef.decode(tokens[i])
        token_aft = tokenizer_aft.encode(sent_i, add_special_tokens=True, max_length=128,
                                         pad_to_max_length=True, return_tensors='pt')
        tokens_aft.append(token_aft)
    return torch.cat(tokens_aft, dim=0)
