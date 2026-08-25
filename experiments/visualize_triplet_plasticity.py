"""
Visualization and analysis of Triplet STDP plasticity.

Usage:
    poetry run python experiments/visualize_triplet_plasticity.py \
        results/triplet_stdp/20260823_135830

The script analyzes:
    - LTP
    - LTD
    - NET ΔW
    - weight distributions
    - firing rates
    - plasticity balance between layers

All plots are saved into:

    <run_dir>/plasticity_plots/
"""

import os
import sys
import json

import torch
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Helpers
# ============================================================

def load_tensor(path):
    """Load a tensor saved by torch.save()."""

    tensor = torch.load(
        path,
        map_location="cpu",
    )

    if isinstance(tensor, torch.Tensor):
        return tensor.float().numpy()

    return np.asarray(tensor, dtype=np.float32)


def load_json(path):
    """Load JSON file."""

    with open(path, "r") as f:
        return json.load(f)


def print_statistics(name, ltp, ltd):
    """Print plasticity statistics."""

    net = ltp - ltd

    print("=" * 60)
    print(name)
    print("=" * 60)

    print("\nLTP")
    print(
        f"  mean : {ltp.mean():+.10e}"
    )
    print(
        f"  std  : {ltp.std():.10e}"
    )
    print(
        f"  min  : {ltp.min():+.10e}"
    )
    print(
        f"  max  : {ltp.max():+.10e}"
    )
    print(
        f"  total: {ltp.sum():+.10e}"
    )

    print("\nLTD")
    print(
        f"  mean : {ltd.mean():+.10e}"
    )
    print(
        f"  std  : {ltd.std():.10e}"
    )
    print(
        f"  min  : {ltd.min():+.10e}"
    )
    print(
        f"  max  : {ltd.max():+.10e}"
    )
    print(
        f"  total: {ltd.sum():+.10e}"
    )

    print("\nNET ΔW")
    print(
        f"  mean : {net.mean():+.10e}"
    )
    print(
        f"  std  : {net.std():.10e}"
    )
    print(
        f"  min  : {net.min():+.10e}"
    )
    print(
        f"  max  : {net.max():+.10e}"
    )
    print(
        f"  total: {net.sum():+.10e}"
    )

    increased = np.mean(net > 1e-10) * 100.0
    decreased = np.mean(net < -1e-10) * 100.0
    near_zero = np.mean(np.abs(net) <= 1e-10) * 100.0

    print("\nNET ΔW distribution")
    print(
        f"  Increased : {increased:.4f}%"
    )
    print(
        f"  Decreased : {decreased:.4f}%"
    )
    print(
        f"  Near zero : {near_zero:.4f}%"
    )

    print()


# ============================================================
# Plot 1
# LTP / LTD / NET distributions
# ============================================================

def plot_plasticity_distributions(
    ltp1,
    ltd1,
    ltp2,
    ltd2,
    output_path,
):
    """Plot distributions of LTP, LTD and NET ΔW."""

    net1 = ltp1 - ltd1
    net2 = ltp2 - ltd2

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
    )

    axes[0].hist(
        ltp1.flatten(),
        bins=100,
        alpha=0.7,
        label="LTP",
    )

    axes[0].hist(
        ltd1.flatten(),
        bins=100,
        alpha=0.7,
        label="LTD",
    )

    axes[0].set_title(
        "Layer 1: LTP and LTD"
    )

    axes[0].set_xlabel(
        "Plasticity magnitude"
    )

    axes[0].set_ylabel(
        "Number of synapses"
    )

    axes[0].legend()

    axes[1].hist(
        net1.flatten(),
        bins=100,
        alpha=0.8,
        label="NET ΔW",
    )

    axes[1].axvline(
        0.0,
        linestyle="--",
    )

    axes[1].set_title(
        "Layer 1: NET ΔW"
    )

    axes[1].set_xlabel(
        "ΔW"
    )

    axes[1].set_ylabel(
        "Number of synapses"
    )

    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
    )
    plt.close()

    # Layer 2

    output_path2 = output_path.replace(
        "layer1",
        "layer2",
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
    )

    axes[0].hist(
        ltp2.flatten(),
        bins=100,
        alpha=0.7,
        label="LTP",
    )

    axes[0].hist(
        ltd2.flatten(),
        bins=100,
        alpha=0.7,
        label="LTD",
    )

    axes[0].set_title(
        "Layer 2: LTP and LTD"
    )

    axes[0].set_xlabel(
        "Plasticity magnitude"
    )

    axes[0].set_ylabel(
        "Number of synapses"
    )

    axes[0].legend()

    axes[1].hist(
        net2.flatten(),
        bins=100,
        alpha=0.8,
        label="NET ΔW",
    )

    axes[1].axvline(
        0.0,
        linestyle="--",
    )

    axes[1].set_title(
        "Layer 2: NET ΔW"
    )

    axes[1].set_xlabel(
        "ΔW"
    )

    axes[1].set_ylabel(
        "Number of synapses"
    )

    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        output_path2,
        dpi=200,
    )
    plt.close()


# ============================================================
# Plot 2
# LTP vs LTD mean
# ============================================================

def plot_ltp_ltd_comparison(
    ltp1,
    ltd1,
    ltp2,
    ltd2,
    output_path,
):
    """Compare average LTP and LTD between layers."""

    ltp_means = [
        ltp1.mean(),
        ltp2.mean(),
    ]

    ltd_means = [
        ltd1.mean(),
        ltd2.mean(),
    ]

    x = np.arange(2)
    width = 0.35

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.bar(
        x - width / 2,
        ltp_means,
        width,
        label="LTP",
    )

    ax.bar(
        x + width / 2,
        ltd_means,
        width,
        label="LTD",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Layer 1",
            "Layer 2",
        ]
    )

    ax.set_ylabel(
        "Mean plasticity"
    )

    ax.set_title(
        "Mean LTP vs LTD"
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


# ============================================================
# Plot 3
# NET ΔW comparison
# ============================================================

def plot_net_comparison(
    ltp1,
    ltd1,
    ltp2,
    ltd2,
    output_path,
):
    """Compare mean NET ΔW between layers."""

    net1 = ltp1 - ltd1
    net2 = ltp2 - ltd2

    values = [
        net1.mean(),
        net2.mean(),
    ]

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    bars = ax.bar(
        [
            "Layer 1",
            "Layer 2",
        ],
        values,
    )

    ax.axhline(
        0.0,
        linestyle="--",
    )

    ax.set_ylabel(
        "Mean NET ΔW"
    )

    ax.set_title(
        "Net Synaptic Plasticity"
    )

    for bar, value in zip(
        bars,
        values,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value,
            f"{value:+.2e}",
            ha="center",
            va="bottom"
            if value >= 0
            else "top",
        )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


# ============================================================
# Plot 4
# NET ΔW heatmaps
# ============================================================

def plot_net_heatmaps(
    ltp1,
    ltd1,
    ltp2,
    ltd2,
    output_path,
):
    """Plot NET ΔW heatmaps."""

    net1 = ltp1 - ltd1
    net2 = ltp2 - ltd2

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 9),
    )

    im1 = axes[0].imshow(
        net1.T,
        aspect="auto",
        interpolation="nearest",
    )

    axes[0].set_title(
        "Layer 1: NET ΔW"
    )

    axes[0].set_xlabel(
        "Input neuron"
    )

    axes[0].set_ylabel(
        "Output neuron"
    )

    fig.colorbar(
        im1,
        ax=axes[0],
        label="ΔW",
    )

    im2 = axes[1].imshow(
        net2.T,
        aspect="auto",
        interpolation="nearest",
    )

    axes[1].set_title(
        "Layer 2: NET ΔW"
    )

    axes[1].set_xlabel(
        "Input neuron"
    )

    axes[1].set_ylabel(
        "Output neuron"
    )

    fig.colorbar(
        im2,
        ax=axes[1],
        label="ΔW",
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


# ============================================================
# Plot 5
# LTP / LTD heatmaps
# ============================================================

def plot_ltp_ltd_heatmaps(
    ltp1,
    ltd1,
    ltp2,
    ltd2,
    output_path,
):
    """Plot LTP and LTD heatmaps."""

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
    )

    data = [
        ltp1,
        ltd1,
        ltp2,
        ltd2,
    ]

    titles = [
        "Layer 1: LTP",
        "Layer 1: LTD",
        "Layer 2: LTP",
        "Layer 2: LTD",
    ]

    for ax, matrix, title in zip(
        axes.flatten(),
        data,
        titles,
    ):

        im = ax.imshow(
            matrix.T,
            aspect="auto",
            interpolation="nearest",
        )

        ax.set_title(title)

        ax.set_xlabel(
            "Input neuron"
        )

        ax.set_ylabel(
            "Output neuron"
        )

        fig.colorbar(
            im,
            ax=ax,
            label="Plasticity",
        )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


# ============================================================
# Plot 6
# Firing rates
# ============================================================

def plot_firing_rates(
    history,
    output_path,
):
    """Plot firing rates."""

    epochs = history["epoch"]

    rate1 = np.asarray(
        history["rate1"]
    ) * 100.0

    rate2 = np.asarray(
        history["rate2"]
    ) * 100.0

    fig, ax = plt.subplots(
        figsize=(9, 6)
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
        "Firing Rate During Triplet STDP"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


# ============================================================
# Plot 7
# Weight statistics
# ============================================================

def plot_weight_statistics(
    history,
    output_path,
):
    """Plot mean weight evolution."""

    epochs = history["epoch"]

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        epochs,
        history["layer1_mean"],
        marker="o",
        label="Layer 1",
    )

    ax.plot(
        epochs,
        history["layer2_mean"],
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
        "Mean Synaptic Weight"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "poetry run python "
            "experiments/visualize_triplet_plasticity.py "
            "<run_directory>"
        )

        sys.exit(1)

    run_dir = sys.argv[1]

    if not os.path.isdir(run_dir):

        print(
            f"ERROR: directory does not exist:"
        )

        print(run_dir)

        sys.exit(1)

    print(
        "Reading results from:"
    )

    print(run_dir)

    print("=" * 60)
    print(
        "TRIPLET STDP PLASTICITY ANALYSIS"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Required files
    # --------------------------------------------------------

    required_files = [
        "ltp_layer1.pt",
        "ltd_layer1.pt",
        "ltp_layer2.pt",
        "ltd_layer2.pt",
        "history.json",
    ]

    for filename in required_files:

        path = os.path.join(
            run_dir,
            filename,
        )

        if not os.path.exists(path):

            print(
                f"\nERROR: missing file:"
            )

            print(path)

            sys.exit(1)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

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

    history = load_json(
        os.path.join(
            run_dir,
            "history.json",
        )
    )

    # --------------------------------------------------------
    # Check shapes
    # --------------------------------------------------------

    print()

    print(
        f"LTP Layer 1 shape: {ltp1.shape}"
    )

    print(
        f"LTD Layer 1 shape: {ltd1.shape}"
    )

    print(
        f"LTP Layer 2 shape: {ltp2.shape}"
    )

    print(
        f"LTD Layer 2 shape: {ltd2.shape}"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()

    print_statistics(
        "LAYER 1",
        ltp1,
        ltd1,
    )

    print_statistics(
        "LAYER 2",
        ltp2,
        ltd2,
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = os.path.join(
        run_dir,
        "plasticity_plots",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------

    print(
        "Generating plots..."
    )

    plot_plasticity_distributions(
        ltp1,
        ltd1,
        ltp2,
        ltd2,
        os.path.join(
            output_dir,
            "plasticity_distributions_layer1.png",
        ),
    )

    plot_ltp_ltd_comparison(
        ltp1,
        ltd1,
        ltp2,
        ltd2,
        os.path.join(
            output_dir,
            "ltp_vs_ltd.png",
        ),
    )

    plot_net_comparison(
        ltp1,
        ltd1,
        ltp2,
        ltd2,
        os.path.join(
            output_dir,
            "net_plasticity.png",
        ),
    )

    plot_net_heatmaps(
        ltp1,
        ltd1,
        ltp2,
        ltd2,
        os.path.join(
            output_dir,
            "net_delta_heatmaps.png",
        ),
    )

    plot_ltp_ltd_heatmaps(
        ltp1,
        ltd1,
        ltp2,
        ltd2,
        os.path.join(
            output_dir,
            "ltp_ltd_heatmaps.png",
        ),
    )

    plot_firing_rates(
        history,
        os.path.join(
            output_dir,
            "firing_rates.png",
        ),
    )

    plot_weight_statistics(
        history,
        os.path.join(
            output_dir,
            "mean_weight_evolution.png",
        ),
    )

    # --------------------------------------------------------
    # Save numerical summary
    # --------------------------------------------------------

    net1 = ltp1 - ltd1
    net2 = ltp2 - ltd2

    summary = {

        "layer1": {
            "ltp_mean": float(ltp1.mean()),
            "ltd_mean": float(ltd1.mean()),
            "net_mean": float(net1.mean()),
            "ltp_total": float(ltp1.sum()),
            "ltd_total": float(ltd1.sum()),
            "net_total": float(net1.sum()),
            "increased_percent": float(
                np.mean(net1 > 1e-10) * 100
            ),
            "decreased_percent": float(
                np.mean(net1 < -1e-10) * 100
            ),
        },

        "layer2": {
            "ltp_mean": float(ltp2.mean()),
            "ltd_mean": float(ltd2.mean()),
            "net_mean": float(net2.mean()),
            "ltp_total": float(ltp2.sum()),
            "ltd_total": float(ltd2.sum()),
            "net_total": float(net2.sum()),
            "increased_percent": float(
                np.mean(net2 > 1e-10) * 100
            ),
            "decreased_percent": float(
                np.mean(net2 < -1e-10) * 100
            ),
        },
    }

    with open(
        os.path.join(
            output_dir,
            "plasticity_summary.json",
        ),
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()

    print("=" * 60)

    print(
        "TRIPLET STDP PLASTICITY ANALYSIS FINISHED"
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