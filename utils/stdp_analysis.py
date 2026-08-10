"""
Analysis and visualization utilities for STDP experiments.

Stores:
- weight statistics
- firing rates
- weight distributions
- weight bounds across epochs
"""

import json
import os

import matplotlib.pyplot as plt
import torch


def create_results_directory(path="results/stdp"):
    """Create directory for STDP results."""
    os.makedirs(path, exist_ok=True)


# ============================================================
# Weight statistics
# ============================================================

def get_weight_statistics(weights):
    """
    Calculate basic statistics of a weight matrix.

    Returns:
        dict with min, max, mean, std, median
        and fractions of weights at the boundaries.
    """

    weights = weights.detach().float()

    w_min = weights.min().item()
    w_max = weights.max().item()

    return {
        "min": w_min,
        "max": w_max,
        "mean": weights.mean().item(),
        "std": weights.std().item(),
        "median": weights.median().item(),

        "zero_fraction": (
            (weights == 0.0).float().mean().item()
        ),

        "max_fraction": (
            (weights == 1.0).float().mean().item()
        ),
    }


# ============================================================
# Collect statistics
# ============================================================

def collect_weight_statistics(model):
    """
    Collect statistics for both STDP layers.
    """

    return {
        "layer1": get_weight_statistics(
            model.layer1.weights
        ),

        "layer2": get_weight_statistics(
            model.layer2.weights
        ),
    }


# ============================================================
# Save JSON
# ============================================================

def save_history(history, path):
    """Save STDP history as JSON."""

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(path, "w") as f:
        json.dump(
            history,
            f,
            indent=4
        )


# ============================================================
# Weight bounds
# ============================================================

def check_weight_bounds(
    model,
    w_min=0.0,
    w_max=1.0,
):
    """
    Check whether weights are inside the allowed range.

    Returns:
        dict
    """

    results = {}

    for name, weights in [
        ("layer1", model.layer1.weights),
        ("layer2", model.layer2.weights),
    ]:

        weights = weights.detach()

        below = (weights < w_min).sum().item()
        above = (weights > w_max).sum().item()

        results[name] = {
            "below_min": below,
            "above_max": above,
            "valid": (
                below == 0 and
                above == 0
            ),
        }

    return results


# ============================================================
# Plot weight bounds
# ============================================================

def plot_weight_bounds(
    history,
    path="results/stdp/weight_bounds.png",
):
    """
    Plot min, mean and max weight values
    for both layers.
    """

    epochs = range(
        1,
        len(history["layer1_mean"]) + 1
    )

    plt.figure(figsize=(9, 5))

    # Layer 1
    plt.plot(
        epochs,
        history["layer1_min"],
        label="Layer 1 min"
    )

    plt.plot(
        epochs,
        history["layer1_mean"],
        label="Layer 1 mean"
    )

    plt.plot(
        epochs,
        history["layer1_max"],
        label="Layer 1 max"
    )

    # Layer 2
    plt.plot(
        epochs,
        history["layer2_min"],
        label="Layer 2 min"
    )

    plt.plot(
        epochs,
        history["layer2_mean"],
        label="Layer 2 mean"
    )

    plt.plot(
        epochs,
        history["layer2_max"],
        label="Layer 2 max"
    )

    # Allowed range
    plt.axhline(
        0.0,
        linestyle="--",
        label="w_min"
    )

    plt.axhline(
        1.0,
        linestyle="--",
        label="w_max"
    )

    plt.xlabel("STDP epoch")
    plt.ylabel("Weight")

    plt.title("STDP Weight Dynamics")

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()


# ============================================================
# Plot firing rates
# ============================================================

def plot_firing_rates(
    history,
    path="results/stdp/firing_rates.png",
):
    """Plot firing rates of both layers."""

    epochs = range(
        1,
        len(history["rate1"]) + 1
    )

    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        history["rate1"],
        label="Layer 1"
    )

    plt.plot(
        epochs,
        history["rate2"],
        label="Layer 2"
    )

    plt.axhline(
        0.05,
        linestyle="--",
        label="Target rate"
    )

    plt.xlabel("STDP epoch")
    plt.ylabel("Firing rate")

    plt.title("STDP Firing Rates")

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()


# ============================================================
# Plot weight distributions
# ============================================================

def plot_weight_distribution(
    model,
    path="results/stdp/weight_distribution.png",
):
    """
    Plot weight distributions for both layers.
    """

    layer1 = (
        model.layer1.weights
        .detach()
        .cpu()
        .flatten()
        .numpy()
    )

    layer2 = (
        model.layer2.weights
        .detach()
        .cpu()
        .flatten()
        .numpy()
    )

    plt.figure(figsize=(9, 5))

    plt.hist(
        layer1,
        bins=50,
        alpha=0.6,
        label="Layer 1"
    )

    plt.hist(
        layer2,
        bins=50,
        alpha=0.6,
        label="Layer 2"
    )

    plt.xlabel("Weight")

    plt.ylabel("Number of synapses")

    plt.title("STDP Weight Distribution")

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()


# ============================================================
# Full analysis
# ============================================================

def analyze_stdp(
    model,
    history,
    results_dir="results/stdp",
):
    """
    Run the complete STDP analysis.
    """

    create_results_directory(results_dir)

    # Save history
    save_history(
        history,
        os.path.join(
            results_dir,
            "history.json"
        )
    )

    # Weight statistics
    statistics = collect_weight_statistics(model)

    save_history(
        statistics,
        os.path.join(
            results_dir,
            "weight_statistics.json"
        )
    )

    # Check bounds
    bounds = check_weight_bounds(
        model,
        w_min=0.0,
        w_max=1.0,
    )

    save_history(
        bounds,
        os.path.join(
            results_dir,
            "weight_bounds.json"
        )
    )

    # Plots
    plot_weight_bounds(
        history,
        os.path.join(
            results_dir,
            "weight_bounds.png"
        )
    )

    plot_firing_rates(
        history,
        os.path.join(
            results_dir,
            "firing_rates.png"
        )
    )

    plot_weight_distribution(
        model,
        os.path.join(
            results_dir,
            "weight_distribution.png"
        )
    )

    print()
    print("=" * 60)
    print("STDP ANALYSIS")
    print("=" * 60)

    print("\nLayer 1:")
    print(statistics["layer1"])

    print("\nLayer 2:")
    print(statistics["layer2"])

    print("\nWeight bounds:")

    for layer, result in bounds.items():

        print(
            f"{layer}: "
            f"{'OK' if result['valid'] else 'OUT OF RANGE'}"
        )

    print("=" * 60)