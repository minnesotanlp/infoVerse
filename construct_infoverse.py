import os
import easydict
import json
import time

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
from src.utils import Logger, set_seed, set_model_path, save_model, cut_input
from src.common import CKPT_PATH, parse_args
from src.scores_src.info import get_infoverse

def main():
    args = parse_args(mode='train')

    ##### Set seed
    set_seed(args)

    ##### Set logs
    log_name = f"{args.dataset}_R{args.data_ratio}_{args.backbone}_{args.train_type}_S{args.seed}"
    args.pre_ckpt = './logs/' + log_name + '/epoch{}.model'.format(args.epochs)

    backbone, tokenizer = load_backbone(args.backbone)
    dataset, _, val_loader, test_loader = get_base_dataset(args.dataset, tokenizer, args.batch_size, args.seed, shuffle=False)
    train_loader = DataLoader(dataset.train_dataset, shuffle=False, drop_last=False, batch_size=args.batch_size, num_workers=4)

    labels_t, labels_v = dataset.train_dataset[:][1][:, 0].numpy(), dataset.val_dataset[:][1][:, 0].numpy()

    args.n_class = dataset.n_classes
    model = Classifier(args.backbone, backbone, args.n_class, args.train_type).cuda()
    
    start = time.time()
    seed_list = [int(item) for item in args.seed_list.split(' ')]
    infoverse = get_infoverse(args, 
                              label_dataset=dataset.train_dataset, 
                              pool_dataset=dataset.train_dataset, 
                              n_epochs=args.epochs, 
                              seeds_list=seed_list, 
                              n_class=args.n_class, 
                              active=False)
    end = time.time()

    # Save the constructed infoverse
    loc = './outputs/{}_{}_infoverse'.format(args.dataset, args.backbone)
    np.save(loc, infoverse)    
    print(f"InfoVerse is successfully constructed (Consumed time: {end - start:.5f} sec)")

if __name__ == "__main__":
    main()
