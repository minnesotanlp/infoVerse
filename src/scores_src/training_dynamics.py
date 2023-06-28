import torch

def avg_conf_variab(ens_sources, n_epochs=-1):
    """
    Args:
        ens_sources['probs']: torch.tensor ([n_epoch, n_samples, n_class])
    Returns:
        (1) average confidence and (2) variability along the whole training epochs
        (https://arxiv.org/abs/2009.10795, EMNLP 20)
    """

    n_samples = ens_sources['probs'].size(1)

    if n_epochs > 0:
        all_conf = ens_sources['probs'][:n_epochs, torch.arange(n_samples), ens_sources['labels']]  # T x N
    else:
        all_conf = ens_sources['probs'][:, torch.arange(n_samples), ens_sources['labels']]  # T x N
    mu = all_conf.mean(dim=0, keepdim=True)
    sigma = ((all_conf - mu) ** 2).mean(dim=0).sqrt()

    return mu.squeeze(0), sigma


def avg_forgetting(ens_sources):
    """
    Args:
        ens_sources['probs']: torch.tensor ([n_epoch, n_samples, n_class])
    Returns:
        (1) forgetting number
        (https://openreview.net/pdf?id=BJlxm30cKm, ICLR 19)
    """

    n_epochs = ens_sources['probs'].size(0)
    n_samples = ens_sources['probs'].size(1)

    # 1. Calculate the correctness for each epoch
    corrects = torch.zeros(n_epochs, n_samples)
    for i in range(n_epochs):
        preds_i = ens_sources['probs'][i]
        preds_i_cls = preds_i.max(dim=1)[1]

        # Correctness of prediction for all samples at i-th epoch
        correct_i = (preds_i_cls == ens_sources['labels']).float()
        corrects[i, :] = correct_i

    # 2. Based on the above results, calculating the forgetting numbers
    forget_score = torch.zeros(n_samples)
    for j in range(1, n_epochs):
        forget_score += (corrects[j] < corrects[j - 1]).float()

    # 3. For samples with wrong predictions among all epochs, giving the highest number
    all_wrong_idx = (corrects.sum(dim=0) == 0)
    forget_score[all_wrong_idx] = n_epochs

    return forget_score

def avg_aum(ens_sources):
    """
    Args:
        ens_sources['logits']: torch.tensor ([n_epoch, n_samples, n_class])
    Returns:
        (1) Area Under Margin (AUM)
        (https://arxiv.org/abs/2001.10528, NeurIPS 20)
    """
    n_epochs = ens_sources['logits'].size(0)
    n_samples = ens_sources['logits'].size(1)

    aum = 0
    for i in range(n_epochs):
        logits_i_epoch = ens_sources['logits'][i].clone()
        logits_i_true = logits_i_epoch[torch.arange(n_samples), ens_sources['labels']].clone()

        logits_i_epoch[torch.arange(n_samples), ens_sources['labels']] = 0
        logits_i_other = logits_i_epoch.max(dim=1)[0]

        aum += (logits_i_true - logits_i_other)

    return aum / n_epochs