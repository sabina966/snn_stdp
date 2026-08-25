"""
Visualization and analysis of Triplet STDP training.

This script does NOT retrain the network.

It reads:

    initial_weights_layer1.pt
    initial_weights_layer2.pt
    final_weights_layer1.pt
    final_weights_layer2.pt
    delta_weights_layer1.pt
    delta_weights_layer2.pt

and creates:

    weight_distributions.png
    delta_distributions.png
    weight_heatmaps.png
    delta_heatmaps.png
    firing_rates.png
    weight_evolution.png
"""

import os
import json
import argparse

import torch
import matplotlib.pyplot as plt


# ============================================================
# Helpers
# ============================================================

def load_json(path):

    with open(path, "r") as f:
        return json.load(f)


def load_tensor(path):

    return torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )


# ============================================================
# Weight distributions
# ============================================================

def plot_weight_distributions(
    initial1,
    final1,
    initial2,
    final2,
    output_path,
):
    """
    Histograms of initial and final weights.
    """

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9, 9),
    )

    # --------------------------------------------------------
    # Layer 1
    # --------------------------------------------------------

    axes[0].hist(
        initial1.flatten().numpy(),
        bins=80,
        alpha=0.5,
        label="Initial",
    )

    axes[0].hist(
        final1.flatten().numpy(),
        bins=80,
        alpha=0.5,
        label="Final",
    )

    axes[0].set_title(
        "Layer 1 weight distribution"
    )

    axes[0].set_xlabel(
        "Weight"
    )

    axes[0].set_ylabel(
        "Number of synapses"
    )

    axes[0].legend()

    axes[0].grid(
        alpha=0.3
    )

    # --------------------------------------------------------
    # Layer 2
    # --------------------------------------------------------

    axes[1].hist(
        initial2.flatten().numpy(),
        bins=80,
        alpha=0.5,
        label="Initial",
    )

    axes[1].hist(
        final2.flatten().numpy(),
        bins=80,
        alpha=0.5,
        label="Final",
    )

    axes[1].set_title(
        "Layer 2 weight distribution"
    )

    axes[1].set_xlabel(
        "Weight"
    )

    axes[1].set_ylabel(
        "Number of synapses"
    )

    axes[1].legend()

    axes[1].grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


# ============================================================
# Delta distributions
# ============================================================

def plot_delta_distributions(
    delta1,
    delta2,
    output_path,
):
    """
    Histograms of ΔW.
    """

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9, 9),
    )

    # --------------------------------------------------------
    # Layer 1
    # --------------------------------------------------------

    axes[0].hist(
        delta1.flatten().numpy(),
        bins=100,
    )

    axes[0].axvline(
        0,
        linestyle="--",
    )

    axes[0].set_title(
        "Layer 1 ΔW distribution"
    )

    axes[0].set_xlabel(
        "ΔW"
    )

    axes[0].set_ylabel(
        "Number of synapses"
    )

    axes[0].grid(
        alpha=0.3
    )

    # --------------------------------------------------------
    # Layer 2
    # --------------------------------------------------------

    axes[1].hist(
        delta2.flatten().numpy(),
        bins=100,
    )

    axes[1].axvline(
        0,
        linestyle="--",
    )

    axes[1].set_title(
        "Layer 2 ΔW distribution"
    )

    axes[1].set_xlabel(
        "ΔW"
    )

    axes[1].set_ylabel(
        "Number of synapses"
    )

    axes[1].grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


# ============================================================
# Weight heatmaps
# ============================================================

def plot_weight_heatmaps(
    initial1,
    final1,
    initial2,
    final2,
    output_path,
):
    """
    Heatmaps of initial and final weights.
    """

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
    )

    # --------------------------------------------------------
    # Layer 1 initial
    # --------------------------------------------------------

    axes[0, 0].imshow(
        initial1.numpy(),
        aspect="auto",
    )

    axes[0, 0].set_title(
        "Layer 1 — initial"
    )

    axes[0, 0].set_xlabel(
        "Neuron"
    )

    axes[0, 0].set_ylabel(
        "Input"
    )

    # --------------------------------------------------------
    # Layer 1 final
    # --------------------------------------------------------

    axes[0, 1].imshow(
        final1.numpy(),
        aspect="auto",
    )

    axes[0, 1].set_title(
        "Layer 1 — final"
    )

    axes[0, 1].set_xlabel(
        "Neuron"
    )

    axes[0, 1].set_ylabel(
        "Input"
    )

    # --------------------------------------------------------
    # Layer 2 initial
    # --------------------------------------------------------

    axes[1, 0].imshow(
        initial2.numpy(),
        aspect="auto",
    )

    axes[1, 0].set_title(
        "Layer 2 — initial"
    )

    axes[1, 0].set_xlabel(
        "Neuron"
    )

    axes[1, 0].set_ylabel(
        "Input"
    )

    # --------------------------------------------------------
    # Layer 2 final
    # --------------------------------------------------------

    axes[1, 1].imshow(
        final2.numpy(),
        aspect="auto",
    )

    axes[1, 1].set_title(
        "Layer 2 — final"
    )

    axes[1, 1].set_xlabel(
        "Neuron"
    )

    axes[1, 1].set_ylabel(
        "Input"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


# ============================================================
# Delta heatmaps
# ============================================================

def plot_delta_heatmaps(
    delta1,
    delta2,
    output_path,
):
    """
    Heatmaps of ΔW.
    """

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
    )

    # --------------------------------------------------------
    # Layer 1
    # --------------------------------------------------------

    vmax1 = delta1.abs().max().item()

    axes[0].imshow(
        delta1.numpy(),
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax1,
        vmax=vmax1,
    )

    axes[0].set_title(
        "Layer 1 — ΔW"
    )

    axes[0].set_xlabel(
        "Neuron"
    )

    axes[0].set_ylabel(
        "Input"
    )

    # --------------------------------------------------------
    # Layer 2
    # --------------------------------------------------------

    vmax2 = delta2.abs().max().item()

    axes[1].imshow(
        delta2.numpy(),
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax2,
        vmax=vmax2,
    )

    axes[1].set_title(
        "Layer 2 — ΔW"
    )

    axes[1].set_xlabel(
        "Neuron"
    )

    axes[1].set_ylabel(
        "Input"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


# ============================================================
# Firing rates
# ============================================================

def plot_firing_rates(
    history,
    output_path,
):
    """
    Plot firing rates.
    """

    epochs = history["epoch"]

    rate1 = [
        value * 100
        for value in history["rate1"]
    ]

    rate2 = [
        value * 100
        for value in history["rate2"]
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        epochs,
        rate1,
        marker="o",
        label="Layer 1",
    )

    ax.plot(
        epochs,
        rate2,
        marker="o",
        label="Layer 2",
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Firing rate (%)"
    )

    ax.set_title(
        "Triplet STDP — firing rate"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


# ============================================================
# Weight evolution
# ============================================================

def plot_weight_evolution(
    history,
    output_path,
):
    """
    Plot mean weight evolution.
    """

    epochs = history["epoch"]

    layer1 = history[
        "layer1_mean"
    ]

    layer2 = history[
        "layer2_mean"
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        epochs,
        layer1,
        marker="o",
        label="Layer 1",
    )

    ax.plot(
        epochs,
        layer2,
        marker="o",
        label="Layer 2",
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Mean weight"
    )

    ax.set_title(
        "Triplet STDP — mean weight evolution"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


# ============================================================
# Numerical summary
# ============================================================

def print_summary(
    initial1,
    initial2,
    final1,
    final2,
    delta1,
    delta2,
):
    """
    Print numerical summary.
    """

    print("=" * 60)
    print("TRIPLET STDP WEIGHT ANALYSIS")
    print("=" * 60)

    for name, initial, final, delta in [
        (
            "LAYER 1",
            initial1,
            final1,
            delta1,
        ),
        (
            "LAYER 2",
            initial2,
            final2,
            delta2,
        ),
    ]:

        total = delta.numel()

        increased = (
            delta > 1e-10
        ).sum().item()

        decreased = (
            delta < -1e-10
        ).sum().item()

        near_zero = total - increased - decreased

        print()
        print(name)

        print(
            f"Initial mean : "
            f"{initial.mean().item():.10f}"
        )

        print(
            f"Final mean   : "
            f"{final.mean().item():.10f}"
        )

        print(
            f"Mean ΔW      : "
            f"{delta.mean().item():+.10e}"
        )

        print(
            f"Median ΔW    : "
            f"{delta.median().item():+.10e}"
        )

        print(
            f"Min ΔW       : "
            f"{delta.min().item():+.10e}"
        )

        print(
            f"Max ΔW       : "
            f"{delta.max().item():+.10e}"
        )

        print(
            f"Std ΔW       : "
            f"{delta.std().item():.10e}"
        )

        print(
            f"Increased    : "
            f"{100 * increased / total:.4f}%"
        )

        print(
            f"Decreased    : "
            f"{100 * decreased / total:.4f}%"
        )

        print(
            f"Near zero    : "
            f"{100 * near_zero / total:.4f}%"
        )

        print(
            f"At zero      : "
            f"{100 * (final == 0).float().mean().item():.4f}%"
        )

        print(
            f"At max       : "
            f"{100 * (final == 0.5).float().mean().item():.4f}%"
        )

    print()
    print("=" * 60)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "run_dir",
        type=str,
        help="Triplet STDP result directory",
    )

    args = parser.parse_args()

    run_dir = args.run_dir

    if not os.path.isdir(run_dir):

        raise FileNotFoundError(
            f"Directory not found: {run_dir}"
        )

    print(
        f"Reading results from:\n{run_dir}"
    )

    # ========================================================
    # Load tensors
    # ========================================================

    initial1 = load_tensor(
        os.path.join(
            run_dir,
            "initial_weights_layer1.pt",
        )
    )

    initial2 = load_tensor(
        os.path.join(
            run_dir,
            "initial_weights_layer2.pt",
        )
    )

    final1 = load_tensor(
        os.path.join(
            run_dir,
            "final_weights_layer1.pt",
        )
    )

    final2 = load_tensor(
        os.path.join(
            run_dir,
            "final_weights_layer2.pt",
        )
    )

    delta1 = load_tensor(
        os.path.join(
            run_dir,
            "delta_weights_layer1.pt",
        )
    )

    delta2 = load_tensor(
        os.path.join(
            run_dir,
            "delta_weights_layer2.pt",
        )
    )

    # ========================================================
    # Load history
    # ========================================================

    history = load_json(
        os.path.join(
            run_dir,
            "history.json",
        )
    )

    # ========================================================
    # Output directory
    # ========================================================

    output_dir = os.path.join(
        run_dir,
        "plots",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # ========================================================
    # Numerical analysis
    # ========================================================

    print_summary(
        initial1,
        initial2,
        final1,
        final2,
        delta1,
        delta2,
    )

    # ========================================================
    # Weight distributions
    # ========================================================

    plot_weight_distributions(
        initial1,
        final1,
        initial2,
        final2,
        os.path.join(
            output_dir,
            "weight_distributions.png",
        ),
    )

    # ========================================================
    # Delta distributions
    # ========================================================

    plot_delta_distributions(
        delta1,
        delta2,
        os.path.join(
            output_dir,
            "delta_distributions.png",
        ),
    )

    # ========================================================
    # Weight heatmaps
    # ========================================================

    plot_weight_heatmaps(
        initial1,
        final1,
        initial2,
        final2,
        os.path.join(
            output_dir,
            "weight_heatmaps.png",
        ),
    )

    # ========================================================
    # Delta heatmaps
    # ========================================================

    plot_delta_heatmaps(
        delta1,
        delta2,
        os.path.join(
            output_dir,
            "delta_heatmaps.png",
        ),
    )

    # ========================================================
    # Firing rates
    # ========================================================

    plot_firing_rates(
        history,
        os.path.join(
            output_dir,
            "firing_rates.png",
        ),
    )

    # ========================================================
    # Weight evolution
    # ========================================================

    plot_weight_evolution(
        history,
        os.path.join(
            output_dir,
            "weight_evolution.png",
        ),
    )

    # ========================================================
    # Finished
    # ========================================================

    print()
    print(
        f"Plots saved to:\n{output_dir}"
    )

    print()
    print("Generated files:")

    for filename in sorted(
        os.listdir(output_dir)
    ):

        print(
            f"  {filename}"
        )


if __name__ == "__main__":
    main()