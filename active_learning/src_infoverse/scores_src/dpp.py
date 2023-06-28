import torch
import numpy as np
import math

def l2_distance(X, X_train):
    num_test = X.shape[0]
    num_train = X_train.shape[0]
    dists = np.tile(np.sum(np.square(X_train), axis=1), (num_test, 1)) \
            + np.tile(np.sum(np.square(X), axis=1), (num_train, 1)).T - 2 * np.dot(X, X_train.T)
    return dists

def gaussian_kernel(feature_matrix):
    # beta = 10 / n_samples -> bandwidth hyperparameter (divnet's choice)
    beta = 1
    dist = l2_distance(feature_matrix, feature_matrix)

    return np.exp(-0.5 * beta * dist)

def label_distance(labels):
    n_samples = len(labels)
    n_class = labels.max() + 1
    onehot = np.zeros((n_samples, n_class))

    onehot[np.arange(n_samples), labels] = 1

    return 1 - 0.5 * l2_distance(onehot, onehot)  # 0: same, 1: different classes

### from https://github.com/laming-chen/fast-map-dpp/blob/master/dpp_test.py
def dpp_greedy(kernel_matrix, max_length, epsilon=1E-11):
    """
    Our proposed fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size), dtype=np.float16)
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)

    while len(selected_items) < max_length:
        k = len(selected_items) - 1

        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)

    if len(selected_items) < max_length:
        print("Selected number of items is {}".format(len(selected_items)))

    return selected_items
