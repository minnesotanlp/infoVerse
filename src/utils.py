import os
import sys
import time
from datetime import datetime
import shutil
import math
import json

import numpy as np
import torch
import random
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Logger(object):
    """Reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514"""
    def __init__(self, fn):
        if not os.path.exists("./logs/"):
            os.mkdir("./logs/")

        logdir = 'logs/' + fn
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        if len(os.listdir(logdir)) != 0:
            # ans = input("log_dir is not empty. All data inside log_dir will be deleted. "
            #                 "Will you proceed [y/N]? ")
            print("log_dir is not empty. original code shows input prompter, but hard-coding for convenience")
            ans = 'y' #TODO: remove it when doing commit or push
            if ans in ['y', 'Y']:
                shutil.rmtree(logdir)
            else:
                exit(1)
        self.set_dir(logdir)

    def set_dir(self, logdir, log_fn='log.txt'):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        self.log_file = open(os.path.join(logdir, log_fn), 'a')

    def log(self, string):
        self.log_file.write('[%s] %s' % (datetime.now(), string) + '\n')
        self.log_file.flush()

        print('[%s] %s' % (datetime.now(), string))
        sys.stdout.flush()

    def log_dirname(self, string):
        self.log_file.write('%s (%s)' % (string, self.logdir) + '\n')
        self.log_file.flush()

        print('%s (%s)' % (string, self.logdir))
        sys.stdout.flush()

def create_tensor_dataset(inputs, labels, index):
    assert len(inputs) == len(labels)
    assert len(inputs) == len(index)

    inputs = torch.stack(inputs)  # (N, T)
    labels = torch.stack(labels).unsqueeze(1)  # (N, 1)
    index = np.array(index)
    index = torch.Tensor(index).long()

    dataset = TensorDataset(inputs, labels, index)

    return dataset

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def set_model_path(args, dataset, epoch, add_str=None):
    # Naming the saving model
    suffix = "_"
    suffix += str(args.train_type)
    suffix += "_epoch" + str(epoch)

    if add_str is not None:
        suffix += add_str

    return dataset.base_path + suffix + '.model'

def save_model(args, model, log_dir, dataset, epoch, add_str=None):
    # Save the model
    if isinstance(model, nn.DataParallel):
        model = model.module

    os.makedirs(log_dir, exist_ok=True)
    model_path = "epoch" + str(epoch) + '.model'
    save_path = os.path.join(log_dir, model_path)
    torch.save(model.state_dict(), save_path)

def cut_input(args, tokens):
    if 'roberta' in args.backbone:
        attention_mask = (tokens != 1).float()
    else:
        attention_mask = (tokens > 0).float()
    max_len = int(torch.max(attention_mask.sum(dim=1)))
    return tokens[:, :max_len], attention_mask[:, :max_len]

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count

def get_raw_data(args, dataset, tokenizer):
    temp_loader = DataLoader(dataset.train_dataset, shuffle=False, drop_last=False, batch_size=1, num_workers=4)

    train_data, train_labels = [], []
    for i, (tokens, labels, _) in enumerate(temp_loader):
        if args.backbone in ['bert', 'albert']:
            num_tokens = (tokens[0] > 0).sum()
        else:
            num_tokens = (tokens[0] != 1).sum()
        orig_sentence = tokenizer.decode(tokens[0, 1:num_tokens - 1])
        train_data.append(orig_sentence)
        train_labels.append(int(labels[0]))

    orig_src_loc = './pre_augment/' + args.dataset + '_' + args.backbone
    with open(orig_src_loc + '_data.txt', "w") as fp:
        json.dump(train_data, fp)

    with open(orig_src_loc + '_label.txt', "w") as fp:
        json.dump(train_labels, fp)

def pruning_dataset(args, datas, infer=False):
    if args.pruning_sample_path is not None:
        pruned_idx = np.load('./pruning/{}_{}_{}_{}_pruning_idx.npy'.format(args.dataset, args.data_ratio, args.backbone, args.pruning_sample_ratio))
    else:
        all_idx = list(range(len(datas)))
        pruning_num = int(args.pruning_sample_ratio * len(datas))
        pruned_idx = random.sample(all_idx, pruning_num)
        remained_idx = list(set(all_idx) - set(pruned_idx))
        assert (len(pruned_idx) + len(remained_idx)) == len(all_idx)
    pruned_dataset = Subset(datas, np.array(remained_idx))
    if infer:
        return DataLoader(pruned_dataset, shuffle=False, drop_last=False, batch_size=args.batch_size, num_workers=4)
    else:
        return DataLoader(pruned_dataset, shuffle=True, drop_last=True, batch_size=args.batch_size, num_workers=4), pruned_idx

