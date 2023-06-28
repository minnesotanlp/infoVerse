import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from sklearn.neighbors import DistanceMetric
from sklearn.utils.extmath import row_norms, stable_cumsum
from tqdm import tqdm


def _kmeans_plusplus(X, n_clusters, dist, random_state, n_local_trials=None):
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)  # random matrix

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly and track index of point
    center_id = random_state.randint(n_samples)
    indices = np.full(n_clusters, -1, dtype=int)  # [-1, ..., -1] array (len=n_clusters)

    centers[0] = X[center_id]
    indices[0] = center_id

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = dist.pairwise(centers[0, np.newaxis], X) ** 2
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = dist.pairwise(X[candidate_ids], X) ** 2

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        centers[c] = X[best_candidate]
        indices[c] = best_candidate

    return centers, indices

def euc_sim(a, b):
    """
      Compute euclidean similarity of 2 sets of vectors

      Parameters:
      a: torch.Tensor, shape: [m, n_features]

      b: torch.Tensor, shape: [n, n_features]
    """
    return - 2 * a @ b.transpose(-2, -1) + (a**2).sum(dim=1)[..., :, None] + (b**2).sum(dim=1)[..., None, :]

def confidence(sources):
    n_samples = len(sources['probs'])
    return -1 * sources['probs'][torch.arange(n_samples), sources['labels']]

def entropy(sources):
    return -1 * (sources['probs'] * torch.log(sources['probs'] + 1e-6)).sum(dim=-1) # T x N

def badge_grads(sources):
    y_onehot = torch.zeros(sources['probs'].shape)
    y_onehot[torch.arange(len(y_onehot)), sources['labels']] = 1

    return (sources['probs'] - y_onehot) * sources['logits']

def badge_grads_norm(sources):
    y_onehot = torch.zeros(sources['probs'].shape)
    y_onehot[torch.arange(len(y_onehot)), sources['labels']] = 1

    return ((sources['probs'] - y_onehot) * sources['logits']).norm(dim=-1)

def badge(sources, n_select):
    # return: selected indices
    g_x = badge_grads(sources)
    dist = DistanceMetric.get_metric('euclidean')
    random_state = np.random.mtrand._rand
    return _kmeans_plusplus(g_x.numpy(), n_select, dist, random_state)

def knn_density_target(sources, K=5):
    sample_distances = euc_sim(sources['penuls'], sources['penuls'])
    sorted_dist, _ = torch.sort(sample_distances, dim=-1)

    return -1 * sorted_dist[:, K]

def surprisal_embed(args, model, loader, p_mask=1.0):
    model.train()

    all_embeds, all_losses = [], []
    n_masks, acc = 0.0, 0.0

    #for i, (tokens, labels, indices) in enumerate(loader):
    for batch in tqdm(loader):
        tokens, labels = batch[0], batch[3]
        batch_size = tokens.size(0)
        attention_mask = (tokens > 0).float() # for bert tokens 
        tokens = tokens.cuda()

        # [CLS] and [SEP] tokens are not masked
        num_tokens = attention_mask.sum(dim=1).long()
        attention_mask[:, 0] = 0
        attention_mask[torch.arange(batch_size), num_tokens - 1] = 0

        mask_p = p_mask * torch.ones(tokens.size()) * attention_mask  # B x L
        mask = torch.bernoulli(mask_p)

        while (mask.sum(dim=-1) == 0).float().sum() > 0:
            mask = torch.bernoulli(mask_p)

            # Inference without masking
        mask_idx = mask.nonzero()
        labels_ssl = -1 * torch.ones(tokens.size()).cuda().long()  # Sampled : 1, not sampled : -1
        labels_ssl[mask_idx[:, 0], mask_idx[:, 1]] = tokens[mask_idx[:, 0], mask_idx[:, 1]]

        with torch.no_grad():
            out_ssl = model(tokens)[0]
            out_ssl = out_ssl.permute(0, 2, 1)

        loss_ssl = F.cross_entropy(out_ssl, labels_ssl, ignore_index=-1, reduction='none').cpu()
        surprisal = loss_ssl / loss_ssl.norm(dim=-1, keepdim=True)

        # Accuracy
        _, pred_ssl = out_ssl.max(dim=1)
        mask_ssl = (labels_ssl != -1).float()
        corrects = (pred_ssl == labels_ssl).cpu().float() * mask
        acc += corrects.sum()
        n_masks += len(mask_idx)

        all_embeds.append(surprisal)
        all_losses.append(loss_ssl.sum(dim=-1) / (num_tokens - 2))

    return torch.cat(all_embeds, dim=0), torch.cat(all_losses, dim=0), acc / n_masks


def surprisal_embed_wino(args, model, loader, p_mask=1.0):
    model.train()

    all_embeds, all_losses = [], []
    n_masks, acc = 0.0, 0.0

    for batch in tqdm(loader):
        tokens, labels = batch[0], batch[3]
        orig_batch_size = tokens.size(0)
        tokens = tokens.reshape(-1, 128)
        batch_size = tokens.size(0)
        attention_mask = (tokens != 1).float()
        tokens = tokens.cuda()

        # [CLS] and [SEP] tokens are not masked
        num_tokens = attention_mask.sum(dim=1).long()
        attention_mask[:, 0] = 0
        attention_mask[torch.arange(batch_size), num_tokens - 1] = 0

        mask_p = p_mask * torch.ones(tokens.size()) * attention_mask  # B x L
        mask = torch.bernoulli(mask_p)

        while (mask.sum(dim=-1) == 0).float().sum() > 0:
            mask = torch.bernoulli(mask_p)

            # Inference without masking
        mask_idx = mask.nonzero()
        labels_ssl = -1 * torch.ones(tokens.size()).cuda().long()  # Sampled : 1, not sampled : -1
        labels_ssl[mask_idx[:, 0], mask_idx[:, 1]] = tokens[mask_idx[:, 0], mask_idx[:, 1]]

        with torch.no_grad():
            out_ssl = model(tokens)[0]
            out_ssl = out_ssl.permute(0, 2, 1)

        loss_ssl = F.cross_entropy(out_ssl, labels_ssl, ignore_index=-1, reduction='none').cpu()
        surprisal = loss_ssl / loss_ssl.norm(dim=-1, keepdim=True)

        # Accuracy
        _, pred_ssl = out_ssl.max(dim=1)
        mask_ssl = (labels_ssl != -1).float()
        corrects = (pred_ssl == labels_ssl).cpu().float() * mask
        acc += corrects.sum()
        n_masks += len(mask_idx)

        surprisal = surprisal.reshape(orig_batch_size, -1)
        all_embeds.append(surprisal)
        loss_all = loss_ssl.sum(dim=-1) / (num_tokens - 2)
        loss_all = loss_all.reshape(orig_batch_size, -1).mean(dim=-1)
        all_losses.append(loss_all)

    return torch.cat(all_embeds, dim=0), torch.cat(all_losses, dim=0), acc / n_masks