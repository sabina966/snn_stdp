"""
Analysis utilities.
"""

import torch


def weight_statistics(weights):
    """
    Calculate statistics of synaptic weights.

    Parameters
    ----------
    weights : Tensor

    Returns
    -------
    dict
    """

    w = weights.detach().cpu()

    stats = {

        "mean": float(w.mean()), # агульны ўзровень ваг

        "std": float(w.std()), # наколькі вагі разышліся

        "min": float(w.min()),

        "max": float(w.max()),

        "median": float(w.median()),

        "dead_fraction": float(
            (w < 0.01).float().mean()
        ),  # колькі сувязяў практычна перасталі ўдзельнічаць

        "strong_fraction": float(
            (w > 0.80).float().mean()
        ), # колькі сувязяў сталі вельмі моцнымі
    }

    return stats

def summarize_history(history):
    """
    Create summary of training history.
    """

    best_epoch = max(
        range(len(history["test_acc"])),
        key=lambda i: history["test_acc"][i]
    )

    summary = {

        "epochs": len(history["train_loss"]),

        "final_train_loss":
            history["train_loss"][-1],

        "final_test_loss":
            history["test_loss"][-1],

        "final_train_accuracy":
            history["train_acc"][-1],

        "final_test_accuracy":
            history["test_acc"][-1],

        "best_test_accuracy":
            history["test_acc"][best_epoch],

        "best_epoch":
            best_epoch + 1,

        "max_layer1_rate":
            max(history["rate1"]),

        "max_layer2_rate":
            max(history["rate2"]),

        "final_layer1_rate":
            history["rate1"][-1],

        "final_layer2_rate":
            history["rate2"][-1],
    }

    return summary