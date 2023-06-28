import torch
from src_infoverse.scores_src.utils import get_features

def mc_dropout_models(args, model, loader, n_ensemble=1):
    ens_sources = []
    model.train()

    for i in range(n_ensemble):
        # Extracting its features
        sources_i = get_features(args, model, loader, mode='train')
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

def el2n_score(ens_sources):
    # Scores from dataset diet (https://arxiv.org/abs/2107.07075, NeurIPS 21 submitted)
    y_onehot = torch.zeros(ens_sources['probs'].shape[1:])
    y_onehot[torch.arange(len(y_onehot)), ens_sources['labels']] = 1
    p_minus_y = ens_sources['probs'] - y_onehot
    return p_minus_y.norm(dim=-1).mean(0)

# Below scores are from (https://openaccess.thecvf.com/content_cvpr_2018/papers/Beluch_The_Power_of_CVPR_2018_paper.pdf)

def ens_max_ent(ens_sources):
    p_ens = ens_sources['probs'].mean(dim=0) # n_samples x n_class
    return -1 * (p_ens * torch.log(p_ens + 1e-6)).sum(dim=-1) # n_samples

def ens_bald(ens_sources):
    z_ent = -1 * (ens_sources['probs'] * torch.log(ens_sources['probs'] + 1e-6)).sum(dim=-1) # n_models x n_samples
    ens_ent = ens_max_ent(ens_sources)
    return ens_ent - z_ent.mean(dim=0)

def ens_varR(ens_sources):
    n_models = ens_sources['logits'].size(0)
    n_samples = ens_sources['logits'].size(1)
    n_class = ens_sources['logits'].size(2)
    all_preds = ens_sources['logits'].max(dim=-1)[1]  # n_models x n_samples

    # 1. Finding average prediction
    n_preds = torch.zeros(n_samples, n_class)
    for t in range(n_models):
        n_preds[torch.arange(n_samples), all_preds[t]] += 1 # Adding weight to the predicted class
    avg_preds = n_preds.max(dim=1)[1]  # Conducting majority voting

    # 2. Calculating the summation of difference
    varR = torch.zeros(n_samples)
    for t in range(n_models):
        varR += (all_preds[t] != avg_preds).long()
    return varR