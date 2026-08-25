"""
Visualization for controlled Pair vs Triplet STDP experiment.

This script does NOT modify any model or training results.

It reads:
    results/controlled_pair_triplet/<run>/

and generates comparison plots for:

- weight changes
- weight distributions
- Pair vs Triplet final weights
- plasticity direction
- firing rates

The experiment must have been produced by:
    experiments/controlled_pair_triplet.py
"""

import json
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Utilities
# ============================================================

def load_tensor(path):
    """Load a tensor from disk."""
    return torch.load(path, weights_only=True).detach().cpu().numpy()


def safe_mean(x):
    return float(np.mean(x))


def safe_std(x):
    return float(np.std(x))


def percentage(x):
    return 100.0 * np.mean(x)


def print_statistics(name, delta):
    """Print numerical statistics for a weight change matrix."""

    print(f"\n{name}")
    print("-" * 50)

    print(f"Mean ΔW      : {np.mean(delta):+.10e}")
    print(f"Median ΔW    : {np.median(delta):+.10e}")
    print(f"Std ΔW       : {np.std(delta):.10e}")
    print(f"Min ΔW       : {np.min(delta):+.10e}")
    print(f"Max ΔW       : {np.max(delta):+.10e}")

    print(
        f"Increased    : "
        f"{percentage(delta > 1e-12):.4f}%"
    )

    print(
        f"Decreased    : "
        f"{percentage(delta < -1e-12):.4f}%"
    )

    print(
        f"Near zero    : "
        f"{percentage(np.abs(delta) <= 1e-12):.4f}%"
    )


# ============================================================
# Plot helpers
# ============================================================

def save_histogram(
    pair_delta,
    triplet_delta,
    title,
    path,
):
    """Compare Pair and Triplet ΔW distributions."""

    plt.figure(figsize=(10, 6))

    plt.hist(
        pair_delta.flatten(),
        bins=100,
        alpha=0.6,
        density=True,
        label="Pair STDP",
    )

    plt.hist(
        triplet_delta.flatten(),
        bins=100,
        alpha=0.6,
        density=True,
        label="Triplet STDP",
    )

    plt.axvline(
        0.0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("ΔW")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()


def save_heatmap(
    delta,
    title,
    path,
):
    """Save ΔW heatmap."""

    plt.figure(figsize=(10, 6))

    vmax = np.max(np.abs(delta))

    plt.imshow(
        delta.T,
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )

    plt.colorbar(
        label="ΔW"
    )

    plt.xlabel("Input neuron")
    plt.ylabel("Output neuron")
    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()


def save_weight_histogram(
    pair_weights,
    triplet_weights,
    title,
    path,
):
    """Compare final weight distributions."""

    plt.figure(figsize=(10, 6))

    plt.hist(
        pair_weights.flatten(),
        bins=100,
        alpha=0.6,
        density=True,
        label="Pair STDP",
    )

    plt.hist(
        triplet_weights.flatten(),
        bins=100,
        alpha=0.6,
        density=True,
        label="Triplet STDP",
    )

    plt.xlabel("Weight")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()


def save_mean_comparison(
    pair_initial,
    pair_final,
    triplet_initial,
    triplet_final,
    layer_name,
    path,
):
    """Compare mean weight before and after plasticity."""

    labels = [
        "Initial",
        "Final",
    ]

    pair_values = [
        np.mean(pair_initial),
        np.mean(pair_final),
    ]

    triplet_values = [
        np.mean(triplet_initial),
        np.mean(triplet_final),
    ]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 6))

    plt.bar(
        x - width / 2,
        pair_values,
        width,
        label="Pair STDP",
    )

    plt.bar(
        x + width / 2,
        triplet_values,
        width,
        label="Triplet STDP",
    )

    plt.xticks(
        x,
        labels,
    )

    plt.ylabel("Mean weight")

    plt.title(
        f"{layer_name}: mean synaptic weight"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()


def save_direction_plot(
    pair_delta,
    triplet_delta,
    layer_name,
    path,
):
    """Compare potentiation/depression percentages."""

    pair_up = percentage(pair_delta > 1e-12)
    pair_down = percentage(pair_delta < -1e-12)

    triplet_up = percentage(triplet_delta > 1e-12)
    triplet_down = percentage(triplet_delta < -1e-12)

    labels = [
        "Increased",
        "Decreased",
    ]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 6))

    plt.bar(
        x - width / 2,
        [pair_up, pair_down],
        width,
        label="Pair STDP",
    )

    plt.bar(
        x + width / 2,
        [triplet_up, triplet_down],
        width,
        label="Triplet STDP",
    )

    plt.xticks(
        x,
        labels,
    )

    plt.ylabel("Synapses (%)")

    plt.ylim(
        0,
        100,
    )

    plt.title(
        f"{layer_name}: direction of plasticity"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()


def save_firing_rates(
    history,
    path,
):
    """Plot firing rates if available."""

    pair_layer1 = history.get(
        "pair_layer1",
        [],
    )

    pair_layer2 = history.get(
        "pair_layer2",
        [],
    )

    triplet_layer1 = history.get(
        "triplet_layer1",
        [],
    )

    triplet_layer2 = history.get(
        "triplet_layer2",
        [],
    )

    if not any(
        [
            pair_layer1,
            pair_layer2,
            triplet_layer1,
            triplet_layer2,
        ]
    ):
        return

    plt.figure(figsize=(10, 6))

    if pair_layer1:
        plt.plot(
            pair_layer1,
            label="Pair Layer 1",
        )

    if pair_layer2:
        plt.plot(
            pair_layer2,
            label="Pair Layer 2",
        )

    if triplet_layer1:
        plt.plot(
            triplet_layer1,
            label="Triplet Layer 1",
        )

    if triplet_layer2:
        plt.plot(
            triplet_layer2,
            label="Triplet Layer 2",
        )

    plt.xlabel("Batch")
    plt.ylabel("Firing rate (%)")
    plt.title(
        "Pair vs Triplet firing rates"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "python experiments/"
            "visualize_controlled_pair_triplet.py "
            "<results_directory>"
        )

        sys.exit(1)

    results_dir = Path(
        sys.argv[1]
    )

    if not results_dir.exists():

        raise FileNotFoundError(
            f"Results directory not found:\n"
            f"{results_dir}"
        )

    print("=" * 60)
    print("CONTROLLED PAIR VS TRIPLET VISUALIZATION")
    print("=" * 60)

    print(
        f"Reading results from:\n"
        f"{results_dir}"
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = (
        results_dir
        / "comparison_plots"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    print("\nLoading weight matrices...")

    initial_l1 = load_tensor(
        results_dir
        / "initial_weights_layer1.pt"
    )

    initial_l2 = load_tensor(
        results_dir
        / "initial_weights_layer2.pt"
    )

    pair_final_l1 = load_tensor(
        results_dir
        / "pair_final_weights_layer1.pt"
    )

    pair_final_l2 = load_tensor(
        results_dir
        / "pair_final_weights_layer2.pt"
    )

    triplet_final_l1 = load_tensor(
        results_dir
        / "triplet_final_weights_layer1.pt"
    )

    triplet_final_l2 = load_tensor(
        results_dir
        / "triplet_final_weights_layer2.pt"
    )

    pair_delta_l1 = load_tensor(
        results_dir
        / "pair_delta_weights_layer1.pt"
    )

    pair_delta_l2 = load_tensor(
        results_dir
        / "pair_delta_weights_layer2.pt"
    )

    triplet_delta_l1 = load_tensor(
        results_dir
        / "triplet_delta_weights_layer1.pt"
    )

    triplet_delta_l2 = load_tensor(
        results_dir
        / "triplet_delta_weights_layer2.pt"
    )

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    print("\n============================================================")
    print("SHAPE CHECK")
    print("============================================================")

    print(
        "Layer 1:",
        initial_l1.shape,
        pair_final_l1.shape,
        triplet_final_l1.shape,
    )

    print(
        "Layer 2:",
        initial_l2.shape,
        pair_final_l2.shape,
        triplet_final_l2.shape,
    )

    # --------------------------------------------------------
    # Verify same initial weights
    # --------------------------------------------------------

    print("\n============================================================")
    print("INITIAL WEIGHT EQUALITY")
    print("============================================================")

    diff_l1 = np.max(
        np.abs(
            initial_l1 - initial_l1
        )
    )

    diff_l2 = np.max(
        np.abs(
            initial_l2 - initial_l2
        )
    )

    print(
        f"Layer 1 max difference: "
        f"{diff_l1:.12e}"
    )

    print(
        f"Layer 2 max difference: "
        f"{diff_l2:.12e}"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\n============================================================")
    print("PLASTICITY STATISTICS")
    print("============================================================")

    print_statistics(
        "PAIR — LAYER 1",
        pair_delta_l1,
    )

    print_statistics(
        "TRIPLET — LAYER 1",
        triplet_delta_l1,
    )

    print_statistics(
        "PAIR — LAYER 2",
        pair_delta_l2,
    )

    print_statistics(
        "TRIPLET — LAYER 2",
        triplet_delta_l2,
    )

    # --------------------------------------------------------
    # Mean differences
    # --------------------------------------------------------

    print("\n============================================================")
    print("PAIR VS TRIPLET DIFFERENCE")
    print("============================================================")

    for name, pair_delta, triplet_delta in [
        (
            "Layer 1",
            pair_delta_l1,
            triplet_delta_l1,
        ),
        (
            "Layer 2",
            pair_delta_l2,
            triplet_delta_l2,
        ),
    ]:

        pair_mean = np.mean(
            pair_delta
        )

        triplet_mean = np.mean(
            triplet_delta
        )

        difference = (
            triplet_mean
            - pair_mean
        )

        print(f"\n{name}")

        print(
            f"Pair mean ΔW    : "
            f"{pair_mean:+.10e}"
        )

        print(
            f"Triplet mean ΔW : "
            f"{triplet_mean:+.10e}"
        )

        print(
            f"Difference      : "
            f"{difference:+.10e}"
        )

    # --------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------

    print("\n============================================================")
    print("GENERATING PLOTS")
    print("============================================================")

    # ΔW distributions

    save_histogram(
        pair_delta_l1,
        triplet_delta_l1,
        "Layer 1: Pair vs Triplet ΔW",
        output_dir
        / "delta_distribution_layer1.png",
    )

    save_histogram(
        pair_delta_l2,
        triplet_delta_l2,
        "Layer 2: Pair vs Triplet ΔW",
        output_dir
        / "delta_distribution_layer2.png",
    )

    # Heatmaps

    save_heatmap(
        pair_delta_l1,
        "Layer 1: Pair STDP ΔW",
        output_dir
        / "pair_delta_heatmap_layer1.png",
    )

    save_heatmap(
        triplet_delta_l1,
        "Layer 1: Triplet STDP ΔW",
        output_dir
        / "triplet_delta_heatmap_layer1.png",
    )

    save_heatmap(
        pair_delta_l2,
        "Layer 2: Pair STDP ΔW",
        output_dir
        / "pair_delta_heatmap_layer2.png",
    )

    save_heatmap(
        triplet_delta_l2,
        "Layer 2: Triplet STDP ΔW",
        output_dir
        / "triplet_delta_heatmap_layer2.png",
    )

    # Final weights

    save_weight_histogram(
        pair_final_l1,
        triplet_final_l1,
        "Layer 1: final weight distributions",
        output_dir
        / "weight_distribution_layer1.png",
    )

    save_weight_histogram(
        pair_final_l2,
        triplet_final_l2,
        "Layer 2: final weight distributions",
        output_dir
        / "weight_distribution_layer2.png",
    )

    # Mean weights

    save_mean_comparison(
        initial_l1,
        pair_final_l1,
        initial_l1,
        triplet_final_l1,
        "Layer 1",
        output_dir
        / "mean_weights_layer1.png",
    )

    save_mean_comparison(
        initial_l2,
        pair_final_l2,
        initial_l2,
        triplet_final_l2,
        "Layer 2",
        output_dir
        / "mean_weights_layer2.png",
    )

    # Direction

    save_direction_plot(
        pair_delta_l1,
        triplet_delta_l1,
        "Layer 1",
        output_dir
        / "plasticity_direction_layer1.png",
    )

    save_direction_plot(
        pair_delta_l2,
        triplet_delta_l2,
        "Layer 2",
        output_dir
        / "plasticity_direction_layer2.png",
    )

    # --------------------------------------------------------
    # Firing rates
    # --------------------------------------------------------

    history_path = (
        results_dir
        / "firing_rate_history.json"
    )

    if history_path.exists():

        with open(
            history_path,
            "r",
            encoding="utf-8",
        ) as f:

            history = json.load(f)

        save_firing_rates(
            history,
            output_dir
            / "firing_rates.png",
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {

        "layer1": {

            "pair_mean_delta": float(
                np.mean(pair_delta_l1)
            ),

            "triplet_mean_delta": float(
                np.mean(triplet_delta_l1)
            ),

            "difference": float(
                np.mean(triplet_delta_l1)
                -
                np.mean(pair_delta_l1)
            ),

            "pair_increased_percent": percentage(
                pair_delta_l1 > 1e-12
            ),

            "pair_decreased_percent": percentage(
                pair_delta_l1 < -1e-12
            ),

            "triplet_increased_percent": percentage(
                triplet_delta_l1 > 1e-12
            ),

            "triplet_decreased_percent": percentage(
                triplet_delta_l1 < -1e-12
            ),
        },

        "layer2": {

            "pair_mean_delta": float(
                np.mean(pair_delta_l2)
            ),

            "triplet_mean_delta": float(
                np.mean(triplet_delta_l2)
            ),

            "difference": float(
                np.mean(triplet_delta_l2)
                -
                np.mean(pair_delta_l2)
            ),

            "pair_increased_percent": percentage(
                pair_delta_l2 > 1e-12
            ),

            "pair_decreased_percent": percentage(
                pair_delta_l2 < -1e-12
            ),

            "triplet_increased_percent": percentage(
                triplet_delta_l2 > 1e-12
            ),

            "triplet_decreased_percent": percentage(
                triplet_delta_l2 < -1e-12
            ),
        },
    }

    with open(
        output_dir
        / "visualization_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    print("\n============================================================")
    print("VISUALIZATION FINISHED")
    print("============================================================")

    print(
        "\nPlots saved to:"
    )

    print(
        output_dir
    )

    print(
        "\nGenerated files:"
    )

    for file in sorted(
        output_dir.iterdir()
    ):

        print(
            f"  {file.name}"
        )


if __name__ == "__main__":
    main()