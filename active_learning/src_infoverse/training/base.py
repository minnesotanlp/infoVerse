import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from training.common import AverageMeter, one_hot, cut_input
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(loader, model, label=False):

    all_preds, all_labels = [], []
    for _, (tokens, labels, _) in enumerate(iter(loader)):
        # Pre-processing
        batch_size = tokens.size(0)
        tokens, _ = cut_input(args, tokens)

        tokens = tokens.to(device)
        labels = labels.to(device)

        labels = labels.squeeze(1)  # (B)

        with torch.no_grad():
            logit = model(tokens)
        preds = torch.max(logit, dim=-1)[1].cpu()

        all_preds.append(preds)
        all_labels.append(labels.cpu())

    if label:
        return torch.cat(all_preds, dim=0), torch.cat(all_labels, dim=0)
    else:
        return torch.cat(all_preds, dim=0)

def train_base(args, loader, model, optimizer, scheduler, epoch=0, logger=None):
    model.train()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['cls_acc'] = AverageMeter()

    criterion = nn.CrossEntropyLoss(reduction='none')

    steps = epoch * len(loader)
    for i, (tokens, labels, _) in enumerate(tqdm(loader)):
        steps += 1
        batch_size = tokens.size(0)

        if args.dataset == 'winogrande':
            tokens = tokens[:, 0, :, :]
            labels = labels - 1
        else:
            tokens, _ = cut_input(args, tokens)

        tokens = tokens.to(device)
        labels = labels.to(device)

        labels = labels.squeeze(1)  # (B)

        out_cls = model(tokens)
        loss = criterion(out_cls, labels).mean()
        (loss / args.grad_accumulation).backward()
        scheduler.step()
        if steps % args.grad_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # cls_acc
        _, pred_cls = out_cls.max(dim=1)
        corrects = (pred_cls == labels).float()
        acc_cls = corrects.sum() / batch_size

        losses['cls'].update(loss.item(), batch_size)
        losses['cls_acc'].update(acc_cls.item(), batch_size)

    msg = '[Epoch %2d] [AccC %.3f] [LossC %.3f]' % (epoch, losses['cls_acc'].average, losses['cls'].average)

    if logger:
        logger.log(msg)
    else:
        print(msg)
