import logging
import os

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

from src.data import (processors, load_and_cache_examples)
from src import setup, sample

from src_infoverse.scores_src.utils import get_features, merge_multiple_models
from src_infoverse.scores_src.others import  surprisal_embed, surprisal_embed_wino, entropy, badge_grads, confidence
from src_infoverse.scores_src.training_dynamics import avg_conf_variab, avg_forgetting, avg_aum
from src_infoverse.scores_src.scoring_mlm_density import get_density_score, get_sentence_embedding, PPCA, compute_nearest_neighbour_distances_cls
from src_infoverse.scores_src.ensembles import mc_dropout_models, el2n_score, ens_max_ent, ens_bald, ens_varR
from src_infoverse.scores_src.dpp import gaussian_kernel, dpp_greedy

from models import load_backbone
from src_infoverse.data.get_dataset import get_base_dataset

logger = logging.getLogger(__name__)


def aggregate(args, features, features_t, alps, alps_t, ens_features_epochs, ens_features_models, mc_ens_features,
              knn_density_s_np):
    all_measurements = []
    name = []

    # Training dynamics
    avg_conf, variab = avg_conf_variab(ens_features_epochs)
    forget_number = avg_forgetting(ens_features_epochs)
    aum = avg_aum(ens_features_epochs)

    all_measurements.append(avg_conf.unsqueeze(0))
    all_measurements.append(variab.unsqueeze(0))
    all_measurements.append(forget_number.unsqueeze(0))
    all_measurements.append(aum.unsqueeze(0))

    name.append('avg_conf') 
    name.append('variab')
    name.append('forget_number')
    name.append('aum')

    # Model Ensemble
    ens_el2n = el2n_score(ens_features_models)
    ens_ent = ens_max_ent(ens_features_models)
    ens_BALD = ens_bald(ens_features_models)
    ens_VAR = ens_varR(ens_features_models)
    ens_avg_conf, ens_variab = avg_conf_variab(ens_features_models)

    all_measurements.append(ens_el2n.unsqueeze(0))
    all_measurements.append(ens_ent.unsqueeze(0))
    all_measurements.append(ens_BALD.unsqueeze(0))
    all_measurements.append(ens_VAR.unsqueeze(0))
    all_measurements.append(ens_avg_conf.unsqueeze(0))
    all_measurements.append(ens_variab.unsqueeze(0))

    name.append('ens_el2n')
    name.append('ens_ent')
    name.append('ens_BALD')
    name.append('ens_VAR')
    name.append('ens_avg_conf')
    name.append('ens_variab')

    # MC Ensemble
    mc_ens_el2n = el2n_score(mc_ens_features)
    mc_ens_ent = ens_max_ent(mc_ens_features)
    mc_ens_BALD = ens_bald(mc_ens_features)
    mc_ens_varR = ens_varR(mc_ens_features)
    mc_ens_avg_conf, mc_ens_variab = avg_conf_variab(mc_ens_features)

    all_measurements.append(mc_ens_el2n.unsqueeze(0))
    all_measurements.append(mc_ens_ent.unsqueeze(0))
    all_measurements.append(mc_ens_BALD.unsqueeze(0))
    all_measurements.append(mc_ens_varR.unsqueeze(0))
    all_measurements.append(mc_ens_avg_conf.unsqueeze(0))
    all_measurements.append(mc_ens_variab.unsqueeze(0))

    name.append('mc_ens_el2n')
    name.append('mc_ens_ent')
    name.append('mc_ens_BALD')
    name.append('mc_ens_VAR')
    name.append('mc_ens_avg_conf')
    name.append('mc_ens_variab')

    # Single Model
    conf = -1 * confidence(features)
    ent = entropy(features)
    badge_embed = badge_grads(features)
    badge_norm = badge_embed.norm(dim=-1)

    args.density_measure = 'nn_relative_dist'
    knn_density_rel = get_density_score(args, features['penuls'].numpy(), features['labels'].numpy(),
                                        features_t['penuls'].numpy(), features_t['labels'].numpy())
    args.density_measure = 'nn_dist'
    knn_density = get_density_score(args, features['penuls'].numpy(), features['labels'].numpy(),
                                    features_t['penuls'].numpy(), features_t['labels'].numpy())

    all_measurements.append(torch.Tensor(knn_density).unsqueeze(0))
    all_measurements.append(torch.Tensor(knn_density_rel).unsqueeze(0))
    all_measurements.append(conf.unsqueeze(0))
    all_measurements.append(ent.unsqueeze(0))
    all_measurements.append(badge_norm.unsqueeze(0))

    name.append('knn_density')  
    name.append('knn_density_rel')
    name.append('conf') 
    name.append('ent')  
    name.append('badge_norm')  

    # Language
    surprisals, likelihood, _ = alps

    sent_density = torch.Tensor(knn_density_s_np).unsqueeze(0)
    likelihoods = -1 * likelihood.unsqueeze(0)

    all_measurements.append(sent_density)  
    all_measurements.append(likelihoods)

    name.append('sent_density')
    name.append('likelihood')

    return all_measurements, name

def get_infoverse(args, label_dataset, pool_dataset, n_epochs, seeds_list, n_class):
    '''
    label_dataset; data of existing labeled samples
    pool_dataset; data of query pool of unlabeled samples
    '''
    label_loader = DataLoader(label_dataset, shuffle=False, drop_last=False, batch_size=args.per_gpu_eval_batch_size)
    pool_loader = DataLoader(pool_dataset, shuffle=False, drop_last=False, batch_size=args.per_gpu_eval_batch_size)

    backbone, tokenizer = load_backbone(args.base_model)
    backbone.to(args.device)

    model, tokenizer, _, _ = setup.load_model(args)

    features_l = get_features(args, model, label_loader)
    features_p = get_features(args, model, pool_loader)

    labels_l = features_l['labels'].numpy()  # True label
    labels_p = features_p['probs'].max(dim=1)[1].numpy()  # Pseudo label

    list_epochs = list(np.arange(n_epochs))
    list_seeds = [seeds_list[0]]
    ens_features_epochs = merge_multiple_models(args, pool_loader, backbone, list_epochs, list_seeds)

    list_epochs = [n_epochs-1]
    list_seeds = seeds_list
    ens_features_models = merge_multiple_models(args, pool_loader, backbone, list_epochs, list_seeds)

    # Caution. Re-initialization is necessary
    model, _, _, _ = setup.load_model(args)
    mc_ens_features = mc_dropout_models(args, model, pool_loader, n_ensemble=len(seeds_list))

    # Label converting
    features_p['labels'] = features_p['logits'].max(dim=1)[1]
    ens_features_epochs['labels'] = features_p['logits'].max(dim=1)[1]
    ens_features_models['labels'] = features_p['logits'].max(dim=1)[1]
    mc_ens_features['labels'] = features_p['logits'].max(dim=1)[1]

    if args.task_name == 'winogrande':
        alps_p = surprisal_embed_wino(args, model, pool_loader)
        alps_l = surprisal_embed_wino(args, model, label_loader)
    else:
        alps_p = surprisal_embed(args, backbone, pool_loader)
        alps_l = surprisal_embed(args, backbone, label_loader)

    backbone, tokenizer = load_backbone('sentence_bert')
    backbone.to(args.device)
    sent_bert_embed_l = get_sentence_embedding(args, backbone, label_loader, aug_src=None, head='SC')[0]
    sent_bert_embed_p = get_sentence_embedding(args, backbone, pool_loader, aug_src=None, head='SC')[0]

    args.density_measure = 'nn_dist'
    knn_density = get_density_score(args, sent_bert_embed_p.numpy(), labels_p, sent_bert_embed_l.numpy(), labels_l)

    res, name = aggregate(args, features_p, features_l, alps_p, alps_l, ens_features_epochs, ens_features_models,
                          mc_ens_features, knn_density)
    return torch.cat(res, dim=0).t(), labels_p


def dpp_sampling(n_query, measurements, labels, scores='density', reduce=False):
    n_sample = len(measurements)
    eps = 5e-4

    measurements_orig = np.array(measurements)
    measurements = (measurements - measurements.mean(axis=0)) / (1e-8 + measurements.std(axis=0))

    # Dimension reduction for removing redundant features
    if reduce:
        info_measures, _ = PPCA(measurements)
    else:
        info_measures = np.array(measurements)

    # Define similarity kernel phi(x_1, x_2)
    similarity = gaussian_kernel(info_measures / np.linalg.norm(info_measures, axis=-1).reshape(-1, 1))

    # To address the case when the same class samples are less than 5
    temp = np.zeros(2)
    for i in range(2):
        temp[i] = (np.array(labels) == i).sum()

    nearest = int(min(max(temp.min() - 2, 0), 5))
    
    # Define score function q(x)
    if scores == 'density':
        scores_bef = -1 * compute_nearest_neighbour_distances_cls(info_measures, labels, info_measures, labels,
                                                                  nearest_k=nearest)
        scores = -1 / (1e-8 + scores_bef)
    elif scores == 'inv':
        scores = compute_nearest_neighbour_distances_cls(info_measures, labels, info_measures, labels,
                                                                  nearest_k=nearest)
    else:
        scores = np.ones(n_sample)
    scores = (scores - scores.min()) / scores.max()
    scores = scores.astype(np.float16)

    dpp_kernel = scores.reshape((n_sample, 1)) * similarity * scores.reshape((1, n_sample))
    selected_idx = dpp_greedy(dpp_kernel + eps * np.eye(n_sample, dtype=np.float16), n_query)

    return selected_idx

def main():
    args = setup.get_args()
    setup.set_seed(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    args.task_name = args.task_name.lower()

    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    
    # First, get already sampled points
    sampled_file = os.path.join(args.model_name_or_path, 'sampled.pt')
    if os.path.isfile(sampled_file):
        sampled = torch.load(sampled_file)
    else:
        sampled = torch.LongTensor([])
    logger.info(f"Already sampled {len(sampled)}")

    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    args.head = sample.sampling_to_head(args.sampling, args.task_name)

    backbone, tokenizer = load_backbone(args.base_model)
    train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
    label_dataset = Subset(train_dataset, np.array(sampled))
    pool_idx = list(set(np.arange(len(train_dataset))) - set(np.array(sampled)))
    pool_dataset = Subset(train_dataset, np.array(pool_idx))

    args.seed_list = list(map(int, args.seed_list.split(" ")))

    info_verse, label_pool = get_infoverse(args, label_dataset=label_dataset, pool_dataset=pool_dataset,
                         n_epochs=int(args.num_train_epochs), seeds_list=args.seed_list, n_class=len(label_list))

    torch.save(info_verse, os.path.join(args.output_dir, 'info_verse.pt'))

    selected = dpp_sampling(args.query_size, info_verse.numpy(), label_pool, scores=args.dpp_sampling)
    selected = list(np.array(pool_idx)[np.array(selected)]) # convert to full-set idx
    queries = torch.cat((sampled, torch.tensor(selected)))

    assert len(queries) == len(queries.unique()), "Duplicates found in sampling"
    assert len(queries) > 0, "Sampling method sampled no queries."

    torch.save(queries, os.path.join(args.output_dir, 'sampled.pt'))
    logger.info(f"Sampled {len(queries)} examples")
    return

if __name__ == "__main__":
    main()
