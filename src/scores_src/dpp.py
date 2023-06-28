import torch
import numpy as np
import math

from src.scores_src import compute_nearest_neighbour_distances_cls

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
def dpp_greedy(kernel_matrix, max_length, save=False, epsilon=1E-11):
    """
    Our proposed fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size), dtype=np.float32)
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)

    print_interval = int(item_size / 10)

    while len(selected_items) < max_length:
        k = len(selected_items) - 1

        if k % print_interval == 0:
            progress = 10 * int(k / print_interval)
            print("Now {} % of processing has been done".format(progress))
            
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
    print("Now {} % of processing has been done".format(100))
                
    if len(selected_items) < max_length:
        print("Selected number of items is {}".format(len(selected_items)))

    return np.array(selected_items)

def dpp_sampling(n_query, measurements, labels, scores='density'):
    n_sample = len(measurements)
    eps = 5e-4
    measurements = np.array(measurements)

    # Normalization
    info_measures = (measurements - measurements.mean(axis=0)) / (1e-8 + measurements.std(axis=0)) 

    # Define similarity kernel phi(x_1, x_2)
    similarity = gaussian_kernel(info_measures / np.linalg.norm(info_measures, axis=-1).reshape(-1, 1))
    
    # Define score function q(x)
    if scores == 'density':
        scores_bef = -1 * compute_nearest_neighbour_distances_cls(info_measures, labels, info_measures, labels, nearest_k=5)
        scores = (-1 / (1e-8 + scores_bef))
    elif scores == 'inv':
        scores = compute_nearest_neighbour_distances_cls(info_measures, labels, info_measures, labels, nearest_k=5)
    else:
        scores = np.ones(n_sample)
    scores = (scores - scores.min()) / scores.max()

    dpp_kernel = scores.reshape((n_sample, 1)) * similarity * scores.reshape((1, n_sample))
    selected_idx = dpp_greedy(dpp_kernel + eps * np.eye(n_sample), n_query)

    return selected_idx
