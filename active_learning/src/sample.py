import numpy as np
import torch
from torch.utils.data import Subset, SequentialSampler, DataLoader
from torch.distributions.categorical import Categorical
from torch.distributions.uniform import Uniform
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, Softmax
from torch.nn.functional import one_hot
import pathlib
import os
from sklearn.cluster import KMeans

def sampling_to_head(sampling, task_name):
    # given [sampling] method, return head of model that is supposed to be used
    head = "lm"
    warmstart = ["badge", "FTbert", "entropy", "bald", "info"]

    for s in warmstart:
        # if sampling needs warmstart method, it needs classification head
        if s in sampling:
            head = 'mc' if task_name == 'winogrande' else "sc"
    return head

def check_model_head(model, sampling, task_name):
    """Check whether [model] is correct for [sampling] method"""
    try:
        model_arch = model.cls.__class__.__name__
    except AttributeError:
        try:
            print(model.config)
            model_arch = model.config.architectures[0]
        except TypeError:
            raise NotImplementedError

    if "MLMHead" in model_arch:
        model_head = "lm"
    elif "SequenceClassification" in model_arch:
        model_head = "sc"
    elif "MultipleChoice" in model_arch:
        model_head = "mc"
    else:
        raise NotImplementedError
    sampling_head = sampling_to_head(sampling, task_name)
    return model_head == sampling_head

def random(inputs, **kwargs):
    """Random sampling by assigning uniformly random scores to all points"""
    scores = Uniform(0,1).sample((inputs["input_ids"].size(0),))
    return scores

def entropy(model, inputs, **kwargs):
    """Maximum entropy sampling by assigning entropy of label distribution for
    example when passed through [model]"""
    logits = model(**inputs)[0]
    categorical = Categorical(logits = logits)
    scores = categorical.entropy()
    return scores

def bald(model, inputs, nsamp=5, **kwargs):
    """Approximate the BALD acquisition function for example when passed through [model]
    reference code: https://github.com/yaringal/acquisition_example/blob/master/acquisition_example.ipynb """
    model.train(True)
    MC_samples = []
    for i in range(nsamp):
        logits = model(**inputs)[0]
        MC_samples.append(torch.softmax(logits, dim=1).cpu())
    MC_samples = torch.stack(MC_samples) # nsamp * batch * class
    expected_entropy = -1 * (MC_samples * torch.log(MC_samples + 1e-6)).sum(dim=-1).mean(dim=0)
    expected_p = MC_samples.mean(dim=0)
    entropy_expected_p = -1 * (expected_p * torch.log(expected_p + 1e-6)).sum(dim=-1)
    scores = entropy_expected_p - expected_entropy

    model.eval()
    return scores

def get_mlm_loss(model, inputs, **kwargs):
    """Obtain masked language modeling loss from [model] for tokens in [inputs].
    Should return batch_size X seq_length tensor."""
    logits = model(**inputs)[1]
    labels = inputs["masked_lm_labels"]
    batch_size, seq_length, vocab_size = logits.size()
    loss_fct = CrossEntropyLoss(reduction='none')
    loss_batched = loss_fct(logits.view(-1, vocab_size), labels.view(-1))
    loss = loss_batched.view(batch_size, seq_length)
    return loss

def badge_gradient(model, inputs, **kwargs):
    """Return the loss gradient with respect to the penultimate layer for BADGE"""
    pooled_output = bert_embedding(model, inputs)
    logits = model.classifier(pooled_output)
    batch_size, num_classes = logits.size()
    softmax = Softmax(dim=1)
    probs = softmax(logits)
    preds = probs.argmax(dim=1)
    preds_oh = one_hot(preds, num_classes=num_classes)
    scales = probs - preds_oh
    grads_3d = torch.einsum('bi,bj->bij', scales, pooled_output)
    grads = grads_3d.view(batch_size, -1)
    return grads

def bert_embedding(model, inputs, **kwargs):
    """Return the [CLS] embedding for each input in [inputs]"""
    inputs.pop("masked_lm_labels", None)
    bert_output = model.bert(**inputs)[1] # This is not working for high version of huggingface due to initialize BertModel with add_pooling_player=False
    return bert_output

SAMPLING = {
    "rand":random,
    "entropy":entropy,
    "badge":badge_gradient,
    "alps": get_mlm_loss,
    "bald": bald
}
def sampling_method(method):
    """Determine function [f] given name of sampling [method] for active learning"""
    if method in SAMPLING:
        f = SAMPLING[method]
    elif "mlm" in method:
        f = get_mlm_loss
    elif "bert" in method:
        f = bert_embedding
    else:
        raise NotImplementedError
    return f

def mask_tokens(inputs, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def get_inputs_for_lm(args, tokenizer, input_ids, attention_mask, token_type_ids=None):
    inputs = {}
    input_ids_cpu = input_ids.cpu().clone()
    input_ids_mask, labels = mask_tokens(input_ids_cpu, tokenizer, args)
    input_ids = input_ids_mask if args.masked else input_ids
    input_ids = input_ids.to(args.device)
    labels = labels.to(args.device)
    inputs["input_ids"] = input_ids
    inputs["masked_lm_labels"] = labels
    inputs["attention_mask"] = attention_mask
    if args.model_type != "distilbert":
        inputs["token_type_ids"] = (
            token_type_ids if args.model_type in ["bert", "xlnet", "albert"] else None
        )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
    return inputs

def batch_scores_or_vectors(batch, args, model, tokenizer):
    """Return scores (or vectors) for data [batch] given the active learning method"""
    model.eval()
    batch = tuple(t.to(args.device) for t in batch)
    inputs = {}
    # mask_tokens() requires CPU input_ids
    if args.head == "lm":
        if args.task_name == 'winogrande':
            num_option = batch[0].shape[1]
            scores_or_vectors = []
            if args.sampling == 'rand':
                inputs['input_ids'] = batch[0][:,0].cpu().clone()
                with torch.no_grad():
                    scores_or_vectors = sampling_method(args.sampling)(model=model, inputs=inputs)
                return scores_or_vectors
            elif args.sampling in ['alps', 'bertKM']:
                for i in range(num_option):
                    input_ids = batch[0][:,i]
                    attention_mask = batch[1][:,i]
                    token_type_ids = (batch[2][:,i] if args.model_type in ["bert", "xlnet", "albert"] else None)
                    inputs = get_inputs_for_lm(args, tokenizer, input_ids, attention_mask, token_type_ids)
                    with torch.no_grad():
                        scores_or_vectors.append(sampling_method(args.sampling)(model=model, inputs=inputs))
                return torch.cat(scores_or_vectors, dim=1)
            else:
                raise NotImplementedError
        else:
            input_ids = batch[0]
            attention_mask = batch[1]
            token_type_ids = (batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None)
            inputs = get_inputs_for_lm(args, tokenizer, input_ids, attention_mask, token_type_ids)

            # input_ids_cpu = batch[0].cpu().clone()
            # input_ids_mask, labels = mask_tokens(input_ids_cpu, tokenizer, args)
            # input_ids = input_ids_mask if args.masked else batch[0]
            # input_ids = input_ids.to(args.device)
            # labels = labels.to(args.device)
            # inputs["input_ids"] = input_ids
            # inputs["masked_lm_labels"] = labels
    elif args.head in ["sc", "mc"]:
        inputs["input_ids"] = batch[0]
        inputs["attention_mask"] = batch[1]
        if args.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
            )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
    else:
        raise NotImplementedError

    with torch.no_grad():
        scores_or_vectors = sampling_method(args.sampling)(model=model, inputs=inputs)
    return scores_or_vectors

def get_scores_or_vectors(eval_dataset, args, model, tokenizer=None):
    # Returns scores or vectors needed for active learning sampling

    assert check_model_head(model, args.sampling, args.task_name), "Model-sampling mismatch"
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)

    for eval_task in eval_task_names:

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        all_scores_or_vectors = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            scores_or_vectors = batch_scores_or_vectors(batch, args, model, tokenizer)

            if all_scores_or_vectors is None:
                all_scores_or_vectors = scores_or_vectors.detach().cpu().numpy()
            else:
                all_scores_or_vectors = np.append(all_scores_or_vectors, scores_or_vectors.detach().cpu().numpy(), axis=0)

    all_scores_or_vectors = torch.tensor(all_scores_or_vectors)
    return all_scores_or_vectors

