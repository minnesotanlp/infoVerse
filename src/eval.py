import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef, f1_score
import scipy.stats as stats

from src.training.common import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cut_input(args, tokens):
    if 'roberta' in args.backbone:
        attention_mask = (tokens != 1).float()
    else:
        attention_mask = (tokens > 0).float()
    max_len = int(torch.max(attention_mask.sum(dim=1)))

    return tokens[:, :max_len]

def acc_k(output, target, ks=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(ks)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    for k in ks:
        correct_k = correct[:k].view(-1).float().sum(0)
        results.append(correct_k.mul_(100.0 / batch_size))
    return results

def test_acc(args, loader, model, logger=None):
    if logger is not None:
        logger.log('Compute test accuracy...')
    model.eval()

    error_top1 = AverageMeter()
    all_preds = []
    all_labels = []

    for i, (tokens, labels, _) in enumerate(loader):
        batch_size = tokens.size(0)

        if args.dataset == 'wino':
            tokens = tokens[:, 0, :, :]
            labels = labels - 1
        else:
            tokens = cut_input(args, tokens)

        tokens = tokens.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            outputs = model(tokens)  # (B, C)

        top1, = acc_k(outputs.data, labels, ks=(1,))
        error_top1.update(top1.item(), batch_size)

        if args.dataset == 'stsb':
            preds = outputs
        else:
            _, preds = outputs.data.cpu().max(1)
        all_preds.append(preds)
        all_labels.append(labels[:, 0].cpu())

    all_preds = torch.cat(all_preds, dim=0).cpu()
    all_labels = torch.cat(all_labels, dim=0)

    # Calculate the F1, MCC
    f1, mcc, p, s = 0, 0, 0, 0

    if args.dataset == 'cola':
        mcc = matthews_corrcoef(all_preds, all_labels)
    elif args.dataset == 'stsb':
        p = stats.pearsonr(all_labels, all_preds[:, 0])[0]
        s = stats.spearmanr(all_labels, all_preds[:, 0])[0]
    elif args.dataset == 'mrpc' or args.dataset == 'qqp':
        f1 = f1_score(all_labels, all_preds)

    return error_top1.average, [100 * mcc, 100 * f1, 100 * p, 100 * s]
