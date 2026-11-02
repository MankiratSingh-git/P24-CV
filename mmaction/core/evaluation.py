"""
Evaluation metrics for MMAction2.
Computes Top-1, Top-5, and Mean Class Accuracy.
"""

import numpy as np
import torch


def top_k_accuracy(scores, labels, topk=(1, 5)):
    """Computes top-k accuracy given predicted logits/scores and ground-truth labels."""
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    maxk = min(max(topk), scores.size(-1))
    batch_size = labels.size(0)

    _, pred = scores.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        if k <= maxk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / max(1, batch_size)).item())
        else:
            res.append(0.0)
    return res
