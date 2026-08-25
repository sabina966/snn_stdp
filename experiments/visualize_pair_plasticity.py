"""
Visualization and analysis of Pure Pair STDP plasticity.

Usage:
    poetry run python experiments/visualize_pair_plasticity.py \
        results/pair_stdp/YYYYMMDD_HHMMSS

Expected files:
    initial_weights_layer1.pt
    initial_weights_layer2.pt
    final_weights_layer1.pt
    final_weights_layer2.pt
    delta_weights_layer1.pt
    delta_weights_layer2.pt
    ltp_layer1.pt
    ltd_layer1.pt
    ltp_layer2.pt
    ltd_layer2.pt
"""

import os
import sys
import json

import torch
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Utilities
# ============================================================

def load_tensor(path):
    """
    Load tensor safely.
    """

    try:
        return torch.load(
            path,
            weights_only=True,
        )
    except TypeError:
        return torch.load(path)


def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
        )


def get_statistics(delta):
    """
    Calculate basic plasticity statistics.
    """

    delta = tensor_to_numpy(delta).ravel()

    near_zero_threshold = 1e-8

    increased = np.sum(delta > near_zero_threshold)
    decreased = np.sum(delta < -near_zero_threshold)
    near_zero = np.sum(
        np.abs(delta) <= near_zero_threshold
    )

    total = len(delta)

    return {
        "mean": float(np.mean(delta)),
        "std": float(np.std(delta)),
        "min": float(np.min(delta)),
        "max": float(np.max(delta)),
        "median": float(np.median(delta)),
        "total": float(np.sum(delta)),
        "increased_percent": float(
            100.0 * increased / total
        ),
        "decreased_percent": float(
            100.0 * decreased / total
        ),
        "near_zero_percent": float(
            100.0 * near_zero / total
        ),
        "at_zero_percent": float(
            100.0 * np.sum(delta == 0) / total
        ),
    }


# ============================================================
# Plot helpers
# ============================================================

def plot_heatmap(
    matrix,
    title,
    path,
    cmap="coolwarm",
    symmetric=False,
):
    """
    Plot matrix as heatmap.
    """

    matrix = tensor_to_numpy(matrix)

    plt.figure(figsize=(10, 6))

    if symmetric:
        limit = np.max(np.abs(matrix))

        if limit == 0:
            limit = 1.0

        plt.imshow(
            matrix.T,
            aspect="auto",
            cmap=cmap,
            vmin=-limit,
            vmax=limit,
        )

    else:
        plt.imshow(
            matrix.T,
            aspect="auto",
            cmap=cmap,
        )

    plt.colorbar()

    plt.title(title)
    plt.xlabel("Input / presynaptic neuron")
    plt.ylabel("Postsynaptic neuron")

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=200,
    )
    plt.close()


def plot_distribution(
    delta,
    title,
    path,
):
    """
    Plot distribution of weight changes.
    """

    values = tensor_to_numpy(delta).ravel()

    plt.figure(figsize=(9, 6))

    plt.hist(
        values,
        bins=100,
    )

    plt.axvline(
        0.0,
        linestyle="--",
    )

    plt.xlabel("ΔW")
    plt.ylabel("Number of synapses")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=200,
    )
    plt.close()


def plot_ltp_ltd(
    ltp,
    ltd,
    title,
    path,
):
    """
    Compare LTP and LTD distributions.
    """

    ltp_values = tensor_to_numpy(ltp).ravel()
    ltd_values = tensor_to_numpy(ltd).ravel()

    plt.figure(figsize=(9, 6))

    plt.hist(
        ltp_values,
        bins=80,
        alpha=0.6,
        label="LTP",
    )

    plt.hist(
        ltd_values,
        bins=80,
        alpha=0.6,
        label="LTD",
    )

    plt.xlabel("Plasticity magnitude")
    plt.ylabel("Number of synapses")
    plt.title(title)
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=200,
    )
    plt.close()


def plot_weight_distribution(
    initial,
    final,
    title,
    path,
):
    """
    Compare initial and final weight distributions.
    """

    initial_values = tensor_to_numpy(
        initial
    ).ravel()

    final_values = tensor_to_numpy(
        final
    ).ravel()

    plt.figure(figsize=(9, 6))

    plt.hist(
        initial_values,
        bins=100,
        alpha=0.5,
        label="Initial",
    )

    plt.hist(
        final_values,
        bins=100,
        alpha=0.5,
        label="Final",
    )

    plt.xlabel("Weight")
    plt.ylabel("Number of synapses")
    plt.title(title)
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=200,
    )
    plt.close()


def plot_mean_weight_evolution(
    initial1,
    final1,
    initial2,
    final2,
    path,
):
    """
    Plot mean weight before and after Pair STDP.
    """

    means1 = [
        float(initial1.mean()),
        float(final1.mean()),
    ]

    means2 = [
        float(initial2.mean()),
        float(final2.mean()),
    ]

    x = [0, 1]

    plt.figure(figsize=(8, 6))

    plt.plot(
        x,
        means1,
        marker="o",
        label="Layer 1",
    )

    plt.plot(
        x,
        means2,
        marker="o",
        label="Layer 2",
    )

    plt.xticks(
        [0, 1],
        ["Initial", "Final"],
    )

    plt.ylabel("Mean weight")
    plt.title("Mean weight evolution — Pair STDP")
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=200,
    )
    plt.close()


def plot_firing_rates(
    path,
    rate1=None,
    rate2=None,
):
    """
    Plot firing rates if available.
    """

    if rate1 is None or rate2 is None:
        return

    plt.figure(figsize=(7, 6))

    plt.bar(
        ["Layer 1", "Layer 2"],
        [
            rate1 * 100.0,
            rate2 * 100.0,
        ],
    )

    plt.ylabel("Firing rate (%)")
    plt.title("Pair STDP firing rates")

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=200,
    )
    plt.close()


# ============================================================
# Main analysis
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "  poetry run python "
            "experiments/visualize_pair_plasticity.py "
            "<results_directory>"
        )

        sys.exit(1)

    run_dir = sys.argv[1]

    if not os.path.isdir(run_dir):

        raise FileNotFoundError(
            f"Results directory not found:\n{run_dir}"
        )

    print(
        "Reading results from:"
    )

    print(run_dir)

    print("=" * 60)
    print("PAIR STDP PLASTICITY ANALYSIS")
    print("=" * 60)

    # ========================================================
    # Output directory
    # ========================================================

    output_dir = os.path.join(
        run_dir,
        "plasticity_plots",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # ========================================================
    # Load weights
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
    # Load LTP / LTD
    # ========================================================

    ltp1 = load_tensor(
        os.path.join(
            run_dir,
            "ltp_layer1.pt",
        )
    )

    ltd1 = load_tensor(
        os.path.join(
            run_dir,
            "ltd_layer1.pt",
        )
    )

    ltp2 = load_tensor(
        os.path.join(
            run_dir,
            "ltp_layer2.pt",
        )
    )

    ltd2 = load_tensor(
        os.path.join(
            run_dir,
            "ltd_layer2.pt",
        )
    )

    print()

    print(
        f"Layer 1 shape: "
        f"{tuple(delta1.shape)}"
    )

    print(
        f"Layer 2 shape: "
        f"{tuple(delta2.shape)}"
    )

    # ========================================================
    # Statistics
    # ========================================================

    stats1 = get_statistics(delta1)
    stats2 = get_statistics(delta2)

    # ========================================================
    # Print Layer 1
    # ========================================================

    print()
    print("=" * 60)
    print("LAYER 1")
    print("=" * 60)

    print(
        f"Initial mean : "
        f"{initial1.mean().item():.10f}"
    )

    print(
        f"Final mean   : "
        f"{final1.mean().item():.10f}"
    )

    print(
        f"Mean ΔW      : "
        f"{stats1['mean']:+.10e}"
    )

    print(
        f"Median ΔW    : "
        f"{stats1['median']:+.10e}"
    )

    print(
        f"Min ΔW       : "
        f"{stats1['min']:+.10e}"
    )

    print(
        f"Max ΔW       : "
        f"{stats1['max']:+.10e}"
    )

    print(
        f"Std ΔW       : "
        f"{stats1['std']:.10e}"
    )

    print(
        f"Increased    : "
        f"{stats1['increased_percent']:.4f}%"
    )

    print(
        f"Decreased    : "
        f"{stats1['decreased_percent']:.4f}%"
    )

    print(
        f"Near zero    : "
        f"{stats1['near_zero_percent']:.4f}%"
    )

    print(
        f"At zero      : "
        f"{stats1['at_zero_percent']:.4f}%"
    )

    # ========================================================
    # Print Layer 2
    # ========================================================

    print()
    print("=" * 60)
    print("LAYER 2")
    print("=" * 60)

    print(
        f"Initial mean : "
        f"{initial2.mean().item():.10f}"
    )

    print(
        f"Final mean   : "
        f"{final2.mean().item():.10f}"
    )

    print(
        f"Mean ΔW      : "
        f"{stats2['mean']:+.10e}"
    )

    print(
        f"Median ΔW    : "
        f"{stats2['median']:+.10e}"
    )

    print(
        f"Min ΔW       : "
        f"{stats2['min']:+.10e}"
    )

    print(
        f"Max ΔW       : "
        f"{stats2['max']:+.10e}"
    )

    print(
        f"Std ΔW       : "
        f"{stats2['std']:.10e}"
    )

    print(
        f"Increased    : "
        f"{stats2['increased_percent']:.4f}%"
    )

    print(
        f"Decreased    : "
        f"{stats2['decreased_percent']:.4f}%"
    )

    print(
        f"Near zero    : "
        f"{stats2['near_zero_percent']:.4f}%"
    )

    print(
        f"At zero      : "
        f"{stats2['at_zero_percent']:.4f}%"
    )

    # ========================================================
    # Firing rates
    # ========================================================

    rate1 = None
    rate2 = None

    summary_path = os.path.join(
        run_dir,
        "summary.json",
    )

    if os.path.exists(summary_path):

        with open(
            summary_path,
            "r",
            encoding="utf-8",
        ) as f:

            summary = json.load(f)

        rate1 = summary.get(
            "final_layer1_rate"
        )

        rate2 = summary.get(
            "final_layer2_rate"
        )

    # ========================================================
    # Save summary
    # ========================================================

    plasticity_summary = {

        "experiment": "pair_stdp",

        "layer1": {
            "initial_mean": float(
                initial1.mean()
            ),
            "final_mean": float(
                final1.mean()
            ),
            **stats1,
        },

        "layer2": {
            "initial_mean": float(
                initial2.mean()
            ),
            "final_mean": float(
                final2.mean()
            ),
            **stats2,
        },

        "firing_rates": {
            "layer1": rate1,
            "layer2": rate2,
        },
    }

    save_json(
        plasticity_summary,
        os.path.join(
            output_dir,
            "plasticity_summary.json",
        ),
    )

    # ========================================================
    # Generate plots
    # ========================================================

    print()
    print(
        "Generating plots..."
    )

    # --------------------------------------------------------
    # 1. ΔW distributions
    # --------------------------------------------------------

    plot_distribution(
        delta1,
        "Pair STDP — Layer 1 ΔW distribution",
        os.path.join(
            output_dir,
            "delta_distribution_layer1.png",
        ),
    )

    plot_distribution(
        delta2,
        "Pair STDP — Layer 2 ΔW distribution",
        os.path.join(
            output_dir,
            "delta_distribution_layer2.png",
        ),
    )

    # --------------------------------------------------------
    # 2. LTP / LTD distributions
    # --------------------------------------------------------

    plot_ltp_ltd(
        ltp1,
        ltd1,
        "Pair STDP — Layer 1 LTP vs LTD",
        os.path.join(
            output_dir,
            "ltp_ltd_distribution_layer1.png",
        ),
    )

    plot_ltp_ltd(
        ltp2,
        ltd2,
        "Pair STDP — Layer 2 LTP vs LTD",
        os.path.join(
            output_dir,
            "ltp_ltd_distribution_layer2.png",
        ),
    )

    # --------------------------------------------------------
    # 3. Weight distributions
    # --------------------------------------------------------

    plot_weight_distribution(
        initial1,
        final1,
        "Pair STDP — Layer 1 weight distribution",
        os.path.join(
            output_dir,
            "weight_distribution_layer1.png",
        ),
    )

    plot_weight_distribution(
        initial2,
        final2,
        "Pair STDP — Layer 2 weight distribution",
        os.path.join(
            output_dir,
            "weight_distribution_layer2.png",
        ),
    )

    # --------------------------------------------------------
    # 4. Weight evolution
    # --------------------------------------------------------

    plot_mean_weight_evolution(
        initial1,
        final1,
        initial2,
        final2,
        os.path.join(
            output_dir,
            "mean_weight_evolution.png",
        ),
    )

    # --------------------------------------------------------
    # 5. Heatmaps of ΔW
    # --------------------------------------------------------

    plot_heatmap(
        delta1,
        "Pair STDP — Layer 1 ΔW",
        os.path.join(
            output_dir,
            "delta_heatmap_layer1.png",
        ),
        symmetric=True,
    )

    plot_heatmap(
        delta2,
        "Pair STDP — Layer 2 ΔW",
        os.path.join(
            output_dir,
            "delta_heatmap_layer2.png",
        ),
        symmetric=True,
    )

    # --------------------------------------------------------
    # 6. LTP heatmaps
    # --------------------------------------------------------

    plot_heatmap(
        ltp1,
        "Pair STDP — Layer 1 LTP",
        os.path.join(
            output_dir,
            "ltp_heatmap_layer1.png",
        ),
        cmap="viridis",
    )

    plot_heatmap(
        ltp2,
        "Pair STDP — Layer 2 LTP",
        os.path.join(
            output_dir,
            "ltp_heatmap_layer2.png",
        ),
        cmap="viridis",
    )

    # --------------------------------------------------------
    # 7. LTD heatmaps
    # --------------------------------------------------------

    plot_heatmap(
        -ltd1,
        "Pair STDP — Layer 1 LTD",
        os.path.join(
            output_dir,
            "ltd_heatmap_layer1.png",
        ),
        cmap="viridis",
    )

    plot_heatmap(
        -ltd2,
        "Pair STDP — Layer 2 LTD",
        os.path.join(
            output_dir,
            "ltd_heatmap_layer2.png",
        ),
        cmap="viridis",
    )

    # --------------------------------------------------------
    # 8. Firing rates
    # --------------------------------------------------------

    plot_firing_rates(
        os.path.join(
            output_dir,
            "firing_rates.png",
        ),
        rate1,
        rate2,
    )

    # ========================================================
    # Finished
    # ========================================================

    print()
    print("=" * 60)
    print(
        "PAIR STDP PLASTICITY ANALYSIS FINISHED"
    )
    print("=" * 60)

    print()
    print(
        "Plots saved to:"
    )

    print(output_dir)

    print()
    print(
        "Generated files:"
    )

    for filename in sorted(
        os.listdir(output_dir)
    ):
        print(
            f"  {filename}"
        )


if __name__ == "__main__":
    main()