import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random
import datetime
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from transformers import get_constant_schedule_with_warmup

import sys
sys.path.append("/home/jaehyung/workspace/infoverse")
from src.eval import test_acc
from src.data import get_base_dataset
from src.models import load_backbone, Classifier
from src.training import train_base
from src.common import CKPT_PATH, parse_args
from src.utils import Logger, set_seed, set_model_path, save_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    args = parse_args(mode='train')

    ##### Set seed
    set_seed(args)

    ##### Set logs
    if args.annotation is not None:
        log_name = f"{args.dataset}_R{args.data_ratio}_{args.backbone}_{args.train_type}_{args.annotation}_S{args.seed}"
    else:
        log_name = f"{args.dataset}_R{args.data_ratio}_{args.backbone}_{args.train_type}_S{args.seed}"
    logger = Logger(log_name)
    log_dir = logger.logdir
    logger.log('Log_name =====> {}'.format(log_name))

    ##### Load models and dataset
    logger.log('Loading pre-trained backbone network... ({})'.format(args.backbone))
    backbone, tokenizer = load_backbone(args.backbone)

    logger.log('Initializing dataset...')
    dataset, train_loader, val_loader, test_loader = get_base_dataset(args.dataset, tokenizer, args.batch_size, args.data_ratio, args.seed)
    if args.annotation is not None:
        anno_dataset = torch.load('./preliminary/data_annotation/' + args.dataset + '_' + args.annotation + '.pt')
        concat_dataset = ConcatDataset([dataset.train_dataset, anno_dataset])
        train_loader = DataLoader(concat_dataset, shuffle=True, drop_last=True, batch_size=args.batch_size, num_workers=4)
    
    logger.log('Initializing model and optimizer...')
    model = Classifier(args.backbone, backbone, dataset.n_classes, args.train_type).to(device)

    if args.pre_ckpt is not None:
        logger.log('Loading from pre-trained model')
        model.load_state_dict(torch.load(args.pre_ckpt))

    # Set optimizer (1) fixed learning rate and (2) no weight decay
    optimizer = optim.Adam(model.parameters(), lr=args.model_lr)
    t_total = len(train_loader) * args.epochs
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps = int(0.06 * t_total)) 

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    logger.log('==========> Start training ({})'.format(args.train_type))
    best_acc, final_acc = 0, 0
    
    for epoch in range(1, args.epochs + 1):
        train_base(args, train_loader, model, optimizer, scheduler, epoch, logger)
        best_acc, final_acc = eval_func(args, model, val_loader, test_loader, logger, best_acc, final_acc, log_dir, dataset)
   
    logger.log('================>>>>>> Final Test Accuracy: {}'.format(final_acc))

def eval_func(args, model, val_loader, test_loader, logger, best_acc, final_acc, log_dir, dataset):
    # other_metric; [mcc, f1, p, s]
    acc, _ = test_acc(args, val_loader, model, logger)

    if acc >= best_acc:
        # Save model
        if args.save_ckpt:
            logger.log('Save model...')
            save_model(args, model, log_dir, dataset, 0)

        t_acc, _ = test_acc(args, test_loader, model, logger)

        # Update test accuracy based on validation performance
        best_acc = acc
        final_acc = t_acc

        logger.log('========== Val Acc ==========')
        logger.log('Val acc: {:.3f}'.format(best_acc))
        logger.log('========== Test Acc ==========')
        logger.log('Test acc: {:.3f}'.format(final_acc))

    return best_acc, final_acc

if __name__ == "__main__":
    main()
