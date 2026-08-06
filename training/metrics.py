"""
Metrics for SNN training.
"""

import torch


def accuracy(logits, labels):
    """
    Compute classification accuracy.
    """

    prediction = logits.argmax(dim=1)

    correct = (prediction == labels).sum().item()

    return correct / labels.size(0)


def firing_rate(spikes):
    """
    Mean spike probability.

    spikes:
        [time, batch, neurons]
    """

    return spikes.float().mean().item()