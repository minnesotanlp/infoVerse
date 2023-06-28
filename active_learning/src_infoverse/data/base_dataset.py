import os
import json
from abc import *

import torch
import csv
from torch.utils.data import TensorDataset
import numpy as np

from datasets import load_dataset

def create_tensor_dataset(inputs, labels, index):
    assert len(inputs) == len(labels)
    assert len(inputs) == len(index)

    inputs = torch.stack(inputs)  # (N, T)
    labels = torch.stack(labels).unsqueeze(1)  # (N, 1)
    index = np.array(index)
    index = torch.Tensor(index).long()

    dataset = TensorDataset(inputs, labels, index)

    return dataset


class BaseDataset(metaclass=ABCMeta):
    def __init__(self, data_name, data_dir, total_class, tokenizer, data_ratio=1.0, seed=0):

        self.data_name = data_name
        self.total_class = total_class
        self.root_dir = os.path.join(data_dir)

        self.tokenizer = tokenizer
        self.data_ratio = data_ratio
        self.seed = seed

        self.n_classes = int(self.total_class)  # Split a given data
        self.class_idx = list(range(self.n_classes))  # all classes
        self.max_class = 1000

        self.n_samples = [100000] * self.n_classes

        if not self._check_exists():
            self._preprocess()

        self.train_dataset = torch.load(self._train_path)
        self.val_dataset = torch.load(self._val_path)
        self.test_dataset = torch.load(self._test_path)

    @property
    def base_path(self):
        try:
            tokenizer_name = self.tokenizer.name
        except AttributeError:
            print("tokenizer doesn't have name variable")
            tokenizer_name = self.tokenizer.name_or_path.split("/")[-1]

        if self.data_ratio < 1.0:
            base_path = '{}_{}_R{:.3f}'.format(self.data_name, tokenizer_name, self.data_ratio)
        else:
            base_path = '{}_{}'.format(self.data_name, tokenizer_name)

        return base_path

    @property
    def _train_path(self):
        return os.path.join(self.root_dir, self.base_path + '_train.pth')

    @property
    def _val_path(self):
        return os.path.join(self.root_dir, self.base_path + '_val.pth')

    @property
    def _test_path(self):
        return os.path.join(self.root_dir, self.base_path + '_test.pth')

    def _check_exists(self):
        if not os.path.exists(self._train_path):
            return False
        elif not os.path.exists(self._val_path):
            return False
        elif not os.path.exists(self._test_path):
            return False
        else:
            return True

    @abstractmethod
    def _preprocess(self):
        pass

    @abstractmethod
    def _load_dataset(self, *args, **kwargs):
        pass


class GLUEDataset(BaseDataset):
    def __init__(self, data_name, data_dir, n_class, tokenizer, data_ratio=1.0, seed=0):
        super(GLUEDataset, self).__init__(data_name, data_dir, n_class, tokenizer, data_ratio, seed)

        self.data_name = data_name

    def _preprocess(self):
        print('Pre-processing GLUE dataset...')
        train_dataset = self._load_dataset('train')

        if self.data_name == 'mnli':
            val_dataset = self._load_dataset('validation_matched')
            test_dataset = self._load_dataset('validation_mismatched')
        else:
            val_dataset = self._load_dataset('validation')
            test_dataset = val_dataset

        # Use the same dataset for validation and test
        torch.save(train_dataset, self._train_path)
        torch.save(val_dataset, self._val_path)
        torch.save(test_dataset, self._test_path)

    def _load_dataset(self, mode='train', raw_text=False):
        assert mode in ['train', 'validation', 'validation_matched', 'validation_mismatched']

        print(self.data_name)
        data_set = load_dataset('glue', self.data_name, split=mode)

        # Get the lists of sentences and their labels.
        inputs, labels, indices = [], [], []

        # Set the number of samples for each class
        if self.data_ratio < 1 and mode == 'train':
            all_labels = np.array(data_set['label'])

            num_samples = np.zeros(self.n_classes)
            for i in range(self.n_classes):
                num_samples[i] = int(self.data_ratio * (all_labels == i).sum())
        else:
            num_samples = 10000000 * np.ones(self.n_classes)

        idx = 0
        for i in range(len(data_set)):
            data_n = data_set[i]

            if self.data_name == 'cola':
                toks = self.tokenizer.encode(data_n['sentence'], add_special_tokens=True, max_length=64,
                                             pad_to_max_length=True, return_tensors='pt')
            elif self.data_name == 'sst2':
                toks = self.tokenizer.encode(data_n['sentence'], add_special_tokens=True, max_length=128,
                                             pad_to_max_length=True, return_tensors='pt')
            else:
                if self.data_name == 'qnli':
                    sent1, sent2 = data_n['question'], data_n['sentence']
                elif self.data_name == 'qqp':
                    sent1, sent2 = data_n['question1'], data_n['question2']
                elif self.data_name == 'mnli':
                    sent1, sent2 = data_n['premise'], data_n['hypothesis']
                else:  # wnli, rte, mrpc, stsb
                    sent1, sent2 = data_n['sentence1'], data_n['sentence2']
                toks = self.tokenizer.encode(sent1, sent2, add_special_tokens=True, max_length=128,
                                             pad_to_max_length=True, return_tensors='pt')

            if self.data_name == 'stsb':
                label = torch.tensor(data_n['label'])
            else:
                label = torch.tensor(data_n['label']).long()
            index = torch.tensor(idx).long()

            if mode == 'train':
                if num_samples[data_n['label']] > 0:
                    inputs.append(toks[0])
                    labels.append(label)
                    indices.append(index)

                    num_samples[data_n['label']] -= 1
            else:
                inputs.append(toks[0])
                labels.append(label)
                indices.append(index)

            idx += 1

        dataset = create_tensor_dataset(inputs, labels, indices)
        return dataset

class WinoDataset(BaseDataset):
    def __init__(self, data_dir, tokenizer, data_ratio=1.0, seed=0):
        super(WinoDataset, self).__init__('wino', data_dir, 2, tokenizer, data_ratio, seed)

    def _preprocess(self):

        train_dataset = self._load_dataset('train')
        val_dataset = self._load_dataset('validation')
        test_dataset = val_dataset

        # Use the same dataset for validation and test
        torch.save(train_dataset, self._train_path)
        torch.save(val_dataset, self._val_path)
        torch.save(test_dataset, self._test_path)

    def _load_dataset(self, mode='train', raw_text=False):
        assert mode in ['train', 'validation', 'test']

        data_set = load_dataset('winogrande', 'winogrande_xl')
        data_set = data_set[mode]

        # Get the lists of sentences and their labels.
        inputs, labels, indices = [], [], []

        # Set the number of samples for each class
        if self.data_ratio < 1 and mode == 'train':
            all_labels = np.array(data_set['label'])

            num_samples = np.zeros(self.n_classes)
            for i in range(self.n_classes):
                num_samples[i] = int(self.data_ratio * (all_labels == i).sum())
        else:
            num_samples = 10000000 * np.ones(self.n_classes)

        indx = 0
        for i in range(len(data_set)):
            if i % 1000 == 0:
                print("Number of processed samples: {}".format(i))
            sentence = data_set['sentence'][i]
            option1 = data_set['option1'][i]
            option2 = data_set['option2'][i]
            answer = data_set['answer'][i]
            conj = "_"

            idx = sentence.index(conj)
            context = sentence[:idx]
            option_str = "_ " + sentence[idx + len(conj):].strip()

            option1 = option_str.replace("_", option1)
            option2 = option_str.replace("_", option2)

            tok1 = self.tokenizer.encode(context + option1, add_special_tokens=True, max_length=128,
                                   pad_to_max_length=True, return_tensors='pt')
            tok2 = self.tokenizer.encode(context + option2, add_special_tokens=True, max_length=128,
                                   pad_to_max_length=True, return_tensors='pt')
            tok = torch.cat([tok1, tok2], dim=0).unsqueeze(0)

            label = torch.tensor(int(answer)).long()
            index = torch.tensor(indx).long()

            if mode == 'train':
                if num_samples[int(answer) - 1] > 0:
                    inputs.append(tok)
                    labels.append(label)
                    indices.append(index)

                    num_samples[int(answer) - 1] -= 1
            else:
                inputs.append(tok)
                labels.append(label)
                indices.append(index)

            indx += 1

        dataset = create_tensor_dataset(inputs, labels, indices)
        return dataset
