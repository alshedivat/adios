"""
Metrics for multi-label classification.
"""
import numpy as np

from sklearn.metrics import f1_score

def f1_measure(data, preds, average='binary'):
    # Compute the scores for each output separately
    f1_scores = {k: float(f1_score(data[k], preds[k], average=average))
                 for k in preds}

    # Concatenate the outputs and compute the overall score
    targets_all = np.hstack([data[k] for k in preds])
    preds_all = np.hstack([preds[k] for k in preds])
    f1_scores['all'] = float(f1_score(targets_all, preds_all, average=average))

    return f1_scores

def hamming_loss(data, preds):
    # Compute the scores for each output separately
    hl = {k: float((data[k] != preds[k]).sum(axis=1).mean()) for k in preds}

    # Concatenate the outputs and compute the overall score
    targets_all = np.hstack([data[k] for k in preds])
    preds_all = np.hstack([preds[k] for k in preds])
    hl['all'] = float((targets_all != preds_all).sum(axis=1).mean())

    return hl

def precision_at_k(data, probs, K):
    P_at_K = {}

    # Compute P@K for every output layer separately
    for k in probs:
        if probs[k].shape[1] >= K:
            idx = probs[k].argsort(axis=1)[:,-K:]
            targets_topk = [data[k][i,idx[i]] for i in xrange(len(idx))]
            P_at_K[k] = float(np.mean(targets_topk))

    # Concatenate the outputs and compute the overall P@K
    targets_all = np.hstack([data[k] for k in probs])
    probs_all = np.hstack([probs[k] for k in probs])

    if probs_all.shape[1] >= K:
        idx = probs_all.argsort(axis=1)[:,-K:]
        targets_topk = [targets_all[i,idx[i]] for i in xrange(len(idx))]
        P_at_K['all'] = float(np.mean(targets_topk))

    return P_at_K
