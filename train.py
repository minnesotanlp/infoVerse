import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random
import datetime
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from tqdm import tqdm

from src.eval import test_acc
from src.data import get_base_dataset
from src.models import load_backbone, Classifier
from src.common import CKPT_PATH, parse_args
from src.utils import Logger, set_seed, set_model_path, save_model, cut_input, AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    args = parse_args(mode='train')

    ##### Set seed
    set_seed(args)

    ##### Set logs
    # Data pruning
    log_name = f"{args.dataset}_R{args.data_ratio}_{args.backbone}_{args.train_type}_S{args.seed}"

    logger = Logger(log_name)
    log_dir = logger.logdir
    logger.log('Log_name =====> {}'.format(log_name))

    ##### Load models and dataset
    logger.log('Loading pre-trained backbone network... ({})'.format(args.backbone))
    backbone, tokenizer = load_backbone(args.backbone)

    logger.log('Initializing dataset...')
    dataset, train_loader, val_loader, test_loader = get_base_dataset(args.dataset, tokenizer, args.batch_size, args.seed)
    
    logger.log('Initializing model and optimizer...')
    if args.dataset == 'wino':
        dataset.n_classes = 1
    model = Classifier(args.backbone, backbone, dataset.n_classes, args.train_type).to(device)

    if args.pre_ckpt is not None:
        logger.log('Loading from pre-trained model')
        model.load_state_dict(torch.load(args.pre_ckpt))

    # Set optimizer (1) fixed learning rate and (2) no weight decay
    optimizer = optim.Adam(model.parameters(), lr=args.model_lr, weight_decay=0)
    t_total = len(train_loader) * args.epochs
    
    logger.log('Lr schedule: Linear')
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(0.06 * t_total), num_training_steps=t_total) 
    
    logger.log('==========> Start training ({})'.format(args.train_type))
    best_acc, final_acc = 0, 0
    
    for epoch in range(1, args.epochs + 1):
        train_base(args, train_loader, model, optimizer, scheduler, epoch, logger)
        best_acc, final_acc = eval_func(args, model, val_loader, test_loader, logger, best_acc, final_acc)

        # Save model
        if args.save_ckpt:
            logger.log('Save model...')
            save_model(args, model, log_dir, dataset, epoch)
   
    logger.log('================>>>>>> Final Test Accuracy: {}'.format(final_acc))

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
        if args.dataset == 'wino':
            tokens = tokens[:, 0, :, :]
            labels = labels - 1
        else:
            tokens, _ = cut_input(args, tokens)

        tokens = tokens.to(device)
        labels = labels.to(device).squeeze(1)

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

    
def eval_func(args, model, val_loader, test_loader, logger, best_acc, final_acc):
    acc, other_metric = test_acc(args, val_loader, model, logger)

    if args.dataset == 'cola':
        metric = other_metric[0]
    else:
        metric = acc

    if metric >= best_acc:
        # As val_data == test_data in GLUE, do not inference it again.
        if args.dataset == 'cola' or args.dataset == 'sst2' or args.dataset == 'qnli':
            t_acc, t_other_metric = acc, other_metric
        else:
            t_acc, t_other_metric = test_acc(args, test_loader, model, logger)

        if args.dataset == 'cola':
            t_metric = t_other_metric[0]
        else:
            t_metric = t_acc

        # Update test accuracy based on validation performance
        best_acc, final_acc = metric, t_metric
        logger.log('========== Val Acc ==========')
        logger.log('Val acc: {:.3f}'.format(best_acc))
        logger.log('========== Test Acc ==========')
        logger.log('Test acc: {:.3f}'.format(final_acc))

    return best_acc, final_acc

if __name__ == "__main__":
    main()
