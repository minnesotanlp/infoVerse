import os
import time
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from src_infoverse.data.base_dataset import GLUEDataset, WinoDataset

def get_base_dataset(data_name, data_dir, tokenizer, batch_size=16, data_ratio=1.0, seed=0, shuffle=True):
    print('Initializing base dataset... (name: {})'.format(data_name))

    # Text Classifications
    if data_name == 'wino':
        dataset = WinoDataset(tokenizer, data_dir, data_ratio, seed)
    else:
        if data_name == 'stsb':
            n_class = 1
        elif data_name == 'mnli':
            n_class = 3
        else:
            n_class = 2
        # GLUE TASKs
        dataset = GLUEDataset(data_name, data_dir, n_class, tokenizer, data_ratio, seed)

    train_loader = DataLoader(dataset.train_dataset, shuffle=shuffle, drop_last=True, batch_size=batch_size, num_workers=4)
    val_loader = DataLoader(dataset.val_dataset, shuffle=False, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(dataset.test_dataset, shuffle=False, batch_size=batch_size, num_workers=4)

    return dataset, train_loader, val_loader, test_loader