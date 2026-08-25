"""
Compare Pair STDP and Triplet STDP experiments.

This script:
- loads initial/final weights;
- loads LTP/LTD matrices;
- calculates net plasticity;
- compares Layer 1 and Layer 2;
- compares firing rates;
- generates comparison plots;
- saves comparison_summary.json.

IMPORTANT:
This script only READS experiment results.
It does NOT modify Pair or Triplet experiment directories.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# ============================================================
# Utilities
# ============================================================

def load_tensor(path):
    """Load tensor safely."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def tensor_to_numpy(tensor):
    """Convert torch tensor to numpy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()

    return np.asarray(tensor)


def load_json(path):
    """Load JSON file."""
    path = Path(path)

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Experiment loading
# ============================================================

def load_experiment(result_dir):
    """
    Load all available Pair/Triplet experiment data.
    """

    result_dir = Path(result_dir)

    if not result_dir.exists():
        raise FileNotFoundError(
            f"Results directory does not exist:\n{result_dir}"
        )

    data = {
        "path": result_dir,
        "initial": {},
        "final": {},
        "delta": {},
        "ltp": {},
        "ltd": {},
        "history": None,
        "summary": None,
    }

    for layer in ["layer1", "layer2"]:

        initial_path = result_dir / f"initial_weights_{layer}.pt"
        final_path = result_dir / f"final_weights_{layer}.pt"
        delta_path = result_dir / f"delta_weights_{layer}.pt"

        ltp_path = result_dir / f"ltp_{layer}.pt"
        ltd_path = result_dir / f"ltd_{layer}.pt"

        if initial_path.exists():
            data["initial"][layer] = tensor_to_numpy(
                load_tensor(initial_path)
            )

        if final_path.exists():
            data["final"][layer] = tensor_to_numpy(
                load_tensor(final_path)
            )

        if delta_path.exists():
            data["delta"][layer] = tensor_to_numpy(
                load_tensor(delta_path)
            )

        if ltp_path.exists():
            data["ltp"][layer] = tensor_to_numpy(
                load_tensor(ltp_path)
            )

        if ltd_path.exists():
            data["ltd"][layer] = tensor_to_numpy(
                load_tensor(ltd_path)
            )

    data["history"] = load_json(result_dir / "history.json")
    data["summary"] = load_json(result_dir / "summary.json")

    return data


# ============================================================
# Statistics
# ============================================================

def plasticity_statistics(delta):
    """
    Calculate statistics for net weight changes.
    """

    delta = np.asarray(delta)

    abs_delta = np.abs(delta)

    # Numerical tolerance
    tolerance = 1e-10

    increased = delta > tolerance
    decreased = delta < -tolerance
    near_zero = abs_delta <= tolerance

    return {
        "mean": float(np.mean(delta)),
        "std": float(np.std(delta)),
        "min": float(np.min(delta)),
        "max": float(np.max(delta)),
        "median": float(np.median(delta)),
        "increased_percent": float(
            100.0 * np.mean(increased)
        ),
        "decreased_percent": float(
            100.0 * np.mean(decreased)
        ),
        "near_zero_percent": float(
            100.0 * np.mean(near_zero)
        ),
        "absolute_mean": float(
            np.mean(abs_delta)
        ),
    }


def ltp_ltd_statistics(ltp, ltd):
    """
    Calculate LTP/LTD statistics.

    LTP is stored as positive magnitude.
    LTD is stored as positive magnitude.
    Net = LTP - LTD.
    """

    ltp = np.asarray(ltp)
    ltd = np.asarray(ltd)

    net = ltp - ltd

    return {
        "ltp_mean": float(np.mean(ltp)),
        "ltp_std": float(np.std(ltp)),
        "ltp_min": float(np.min(ltp)),
        "ltp_max": float(np.max(ltp)),
        "ltp_total": float(np.sum(ltp)),

        "ltd_mean": float(np.mean(ltd)),
        "ltd_std": float(np.std(ltd)),
        "ltd_min": float(np.min(ltd)),
        "ltd_max": float(np.max(ltd)),
        "ltd_total": float(np.sum(ltd)),

        "net_mean": float(np.mean(net)),
        "net_std": float(np.std(net)),
        "net_min": float(np.min(net)),
        "net_max": float(np.max(net)),
        "net_total": float(np.sum(net)),

        "ltp_ltd_ratio": float(
            np.sum(ltp) / max(np.sum(ltd), 1e-30)
        ),
    }


def weight_statistics(initial, final):
    """
    Calculate weight statistics before and after learning.
    """

    initial = np.asarray(initial)
    final = np.asarray(final)

    delta = final - initial

    return {
        "initial_mean": float(np.mean(initial)),
        "final_mean": float(np.mean(final)),

        "initial_std": float(np.std(initial)),
        "final_std": float(np.std(final)),

        "initial_min": float(np.min(initial)),
        "final_min": float(np.min(final)),

        "initial_max": float(np.max(initial)),
        "final_max": float(np.max(final)),

        "mean_delta": float(np.mean(delta)),
        "median_delta": float(np.median(delta)),
    }


# ============================================================
# Firing rates
# ============================================================

def extract_firing_rates(experiment):
    """
    Extract final firing rates from history.json.
    """

    history = experiment.get("history")

    if history is None:
        return None

    rate1 = history.get("rate1")
    rate2 = history.get("rate2")

    if rate1 is None or rate2 is None:
        return None

    if len(rate1) == 0 or len(rate2) == 0:
        return None

    return {
        "layer1": float(rate1[-1]),
        "layer2": float(rate2[-1]),
    }


# ============================================================
# Plot helpers
# ============================================================

def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# Plot: mean weight comparison
# ============================================================

def plot_mean_weights(pair, triplet, output_dir):

    layers = ["layer1", "layer2"]

    pair_values = []
    triplet_values = []

    for layer in layers:

        pair_values.append(
            np.mean(pair["final"][layer])
        )

        triplet_values.append(
            np.mean(triplet["final"][layer])
        )

    x = np.arange(len(layers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        x - width / 2,
        pair_values,
        width,
        label="Pair STDP",
    )

    ax.bar(
        x + width / 2,
        triplet_values,
        width,
        label="Triplet STDP",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Layer 1", "Layer 2"]
    )

    ax.set_ylabel("Mean synaptic weight")
    ax.set_title(
        "Final mean synaptic weights"
    )

    ax.legend()

    save_figure(
        fig,
        output_dir / "mean_weights_comparison.png",
    )


# ============================================================
# Plot: net plasticity
# ============================================================

def plot_net_plasticity(pair, triplet, output_dir):

    layers = ["layer1", "layer2"]

    pair_values = []
    triplet_values = []

    for layer in layers:

        pair_delta = (
            pair["final"][layer]
            - pair["initial"][layer]
        )

        triplet_delta = (
            triplet["final"][layer]
            - triplet["initial"][layer]
        )

        pair_values.append(
            np.mean(pair_delta)
        )

        triplet_values.append(
            np.mean(triplet_delta)
        )

    x = np.arange(len(layers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.bar(
        x - width / 2,
        pair_values,
        width,
        label="Pair STDP",
    )

    ax.bar(
        x + width / 2,
        triplet_values,
        width,
        label="Triplet STDP",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Layer 1", "Layer 2"]
    )

    ax.set_ylabel("Mean ΔW")
    ax.set_title(
        "Net synaptic plasticity"
    )

    ax.legend()

    save_figure(
        fig,
        output_dir / "net_plasticity_comparison.png",
    )


# ============================================================
# Plot: increased / decreased
# ============================================================

def plot_direction_comparison(
    pair,
    triplet,
    output_dir,
):

    layers = ["layer1", "layer2"]

    pair_inc = []
    pair_dec = []

    triplet_inc = []
    triplet_dec = []

    for layer in layers:

        pair_delta = (
            pair["final"][layer]
            - pair["initial"][layer]
        )

        triplet_delta = (
            triplet["final"][layer]
            - triplet["initial"][layer]
        )

        pair_stats = plasticity_statistics(
            pair_delta
        )

        triplet_stats = plasticity_statistics(
            triplet_delta
        )

        pair_inc.append(
            pair_stats["increased_percent"]
        )

        pair_dec.append(
            pair_stats["decreased_percent"]
        )

        triplet_inc.append(
            triplet_stats["increased_percent"]
        )

        triplet_dec.append(
            triplet_stats["decreased_percent"]
        )

    x = np.arange(len(layers))
    width = 0.20

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x - 1.5 * width,
        pair_inc,
        width,
        label="Pair increased",
    )

    ax.bar(
        x - 0.5 * width,
        pair_dec,
        width,
        label="Pair decreased",
    )

    ax.bar(
        x + 0.5 * width,
        triplet_inc,
        width,
        label="Triplet increased",
    )

    ax.bar(
        x + 1.5 * width,
        triplet_dec,
        width,
        label="Triplet decreased",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Layer 1", "Layer 2"]
    )

    ax.set_ylabel("Synapses (%)")
    ax.set_title(
        "Direction of synaptic changes"
    )

    ax.legend()

    save_figure(
        fig,
        output_dir / "plasticity_direction_comparison.png",
    )


# ============================================================
# Plot: LTP vs LTD
# ============================================================

def plot_ltp_ltd(
    pair,
    triplet,
    output_dir,
):

    layers = ["layer1", "layer2"]

    pair_ltp = []
    pair_ltd = []

    triplet_ltp = []
    triplet_ltd = []

    for layer in layers:

        pair_stats = ltp_ltd_statistics(
            pair["ltp"][layer],
            pair["ltd"][layer],
        )

        triplet_stats = ltp_ltd_statistics(
            triplet["ltp"][layer],
            triplet["ltd"][layer],
        )

        pair_ltp.append(
            pair_stats["ltp_mean"]
        )

        pair_ltd.append(
            pair_stats["ltd_mean"]
        )

        triplet_ltp.append(
            triplet_stats["ltp_mean"]
        )

        triplet_ltd.append(
            triplet_stats["ltd_mean"]
        )

    x = np.arange(len(layers))
    width = 0.20

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x - 1.5 * width,
        pair_ltp,
        width,
        label="Pair LTP",
    )

    ax.bar(
        x - 0.5 * width,
        pair_ltd,
        width,
        label="Pair LTD",
    )

    ax.bar(
        x + 0.5 * width,
        triplet_ltp,
        width,
        label="Triplet LTP",
    )

    ax.bar(
        x + 1.5 * width,
        triplet_ltd,
        width,
        label="Triplet LTD",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Layer 1", "Layer 2"]
    )

    ax.set_ylabel("Mean plasticity magnitude")
    ax.set_title(
        "LTP and LTD comparison"
    )

    ax.legend()

    save_figure(
        fig,
        output_dir / "ltp_ltd_comparison.png",
    )


# ============================================================
# Plot: firing rates
# ============================================================

def plot_firing_rates(
    pair_rates,
    triplet_rates,
    output_dir,
):

    if pair_rates is None or triplet_rates is None:
        return

    layers = ["layer1", "layer2"]

    pair_values = [
        pair_rates["layer1"] * 100,
        pair_rates["layer2"] * 100,
    ]

    triplet_values = [
        triplet_rates["layer1"] * 100,
        triplet_rates["layer2"] * 100,
    ]

    x = np.arange(len(layers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        x - width / 2,
        pair_values,
        width,
        label="Pair STDP",
    )

    ax.bar(
        x + width / 2,
        triplet_values,
        width,
        label="Triplet STDP",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Layer 1", "Layer 2"]
    )

    ax.set_ylabel("Firing rate (%)")
    ax.set_title(
        "Final firing rates"
    )

    ax.legend()

    save_figure(
        fig,
        output_dir / "firing_rate_comparison.png",
    )


# ============================================================
# Plot: delta distributions
# ============================================================

def plot_delta_distributions(
    pair,
    triplet,
    output_dir,
):

    for layer in ["layer1", "layer2"]:

        pair_delta = (
            pair["final"][layer]
            - pair["initial"][layer]
        )

        triplet_delta = (
            triplet["final"][layer]
            - triplet["initial"][layer]
        )

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.hist(
            pair_delta.flatten(),
            bins=100,
            alpha=0.6,
            label="Pair STDP",
        )

        ax.hist(
            triplet_delta.flatten(),
            bins=100,
            alpha=0.6,
            label="Triplet STDP",
        )

        ax.axvline(
            0,
            linewidth=1,
        )

        ax.set_xlabel("ΔW")
        ax.set_ylabel("Number of synapses")

        ax.set_title(
            f"Weight-change distribution — {layer}"
        )

        ax.legend()

        save_figure(
            fig,
            output_dir / f"delta_distribution_{layer}.png",
        )


# ============================================================
# Main comparison
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare Pair STDP and Triplet STDP "
            "experiments."
        )
    )

    parser.add_argument(
        "pair_dir",
        type=str,
        help="Pair STDP results directory",
    )

    parser.add_argument(
        "triplet_dir",
        type=str,
        help="Triplet STDP results directory",
    )

    args = parser.parse_args()

    pair_dir = Path(args.pair_dir)
    triplet_dir = Path(args.triplet_dir)

    print("=" * 60)
    print("PAIR VS TRIPLET STDP COMPARISON")
    print("=" * 60)

    print("Pair results:")
    print(pair_dir)

    print()

    print("Triplet results:")
    print(triplet_dir)

    print()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    pair = load_experiment(pair_dir)
    triplet = load_experiment(triplet_dir)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    required = [
        "layer1",
        "layer2",
    ]

    for name, experiment in [
        ("Pair", pair),
        ("Triplet", triplet),
    ]:

        for layer in required:

            if layer not in experiment["initial"]:
                raise RuntimeError(
                    f"{name}: missing initial weights "
                    f"for {layer}"
                )

            if layer not in experiment["final"]:
                raise RuntimeError(
                    f"{name}: missing final weights "
                    f"for {layer}"
                )

            if layer not in experiment["ltp"]:
                raise RuntimeError(
                    f"{name}: missing LTP matrix "
                    f"for {layer}"
                )

            if layer not in experiment["ltd"]:
                raise RuntimeError(
                    f"{name}: missing LTD matrix "
                    f"for {layer}"
                )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = (
        pair_dir.parent
        / "pair_vs_triplet_comparison"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    comparison = {
        "pair_results": str(pair_dir),
        "triplet_results": str(triplet_dir),
        "layers": {},
    }

    print("=" * 60)
    print("NUMERICAL COMPARISON")
    print("=" * 60)

    for layer in ["layer1", "layer2"]:

        print()
        print("=" * 60)
        print(layer.upper())
        print("=" * 60)

        # ----------------------------------------------------
        # Pair
        # ----------------------------------------------------

        pair_initial = pair["initial"][layer]
        pair_final = pair["final"][layer]

        pair_delta = (
            pair_final
            - pair_initial
        )

        pair_weight_stats = weight_statistics(
            pair_initial,
            pair_final,
        )

        pair_plasticity = plasticity_statistics(
            pair_delta
        )

        pair_ltp_ltd = ltp_ltd_statistics(
            pair["ltp"][layer],
            pair["ltd"][layer],
        )

        # ----------------------------------------------------
        # Triplet
        # ----------------------------------------------------

        triplet_initial = triplet["initial"][layer]
        triplet_final = triplet["final"][layer]

        triplet_delta = (
            triplet_final
            - triplet_initial
        )

        triplet_weight_stats = weight_statistics(
            triplet_initial,
            triplet_final,
        )

        triplet_plasticity = plasticity_statistics(
            triplet_delta
        )

        triplet_ltp_ltd = ltp_ltd_statistics(
            triplet["ltp"][layer],
            triplet["ltd"][layer],
        )

        # ----------------------------------------------------
        # Print Pair
        # ----------------------------------------------------

        print()
        print("PAIR STDP")
        print("-" * 40)

        print(
            f"Initial mean : "
            f"{pair_weight_stats['initial_mean']:.10f}"
        )

        print(
            f"Final mean   : "
            f"{pair_weight_stats['final_mean']:.10f}"
        )

        print(
            f"Mean ΔW      : "
            f"{pair_plasticity['mean']:+.10e}"
        )

        print(
            f"Increased    : "
            f"{pair_plasticity['increased_percent']:.4f}%"
        )

        print(
            f"Decreased    : "
            f"{pair_plasticity['decreased_percent']:.4f}%"
        )

        print(
            f"LTP mean     : "
            f"{pair_ltp_ltd['ltp_mean']:.10e}"
        )

        print(
            f"LTD mean     : "
            f"{pair_ltp_ltd['ltd_mean']:.10e}"
        )

        print(
            f"LTP total    : "
            f"{pair_ltp_ltd['ltp_total']:.10e}"
        )

        print(
            f"LTD total    : "
            f"{pair_ltp_ltd['ltd_total']:.10e}"
        )

        # ----------------------------------------------------
        # Print Triplet
        # ----------------------------------------------------

        print()
        print("TRIPLET STDP")
        print("-" * 40)

        print(
            f"Initial mean : "
            f"{triplet_weight_stats['initial_mean']:.10f}"
        )

        print(
            f"Final mean   : "
            f"{triplet_weight_stats['final_mean']:.10f}"
        )

        print(
            f"Mean ΔW      : "
            f"{triplet_plasticity['mean']:+.10e}"
        )

        print(
            f"Increased    : "
            f"{triplet_plasticity['increased_percent']:.4f}%"
        )

        print(
            f"Decreased    : "
            f"{triplet_plasticity['decreased_percent']:.4f}%"
        )

        print(
            f"LTP mean     : "
            f"{triplet_ltp_ltd['ltp_mean']:.10e}"
        )

        print(
            f"LTD mean     : "
            f"{triplet_ltp_ltd['ltd_mean']:.10e}"
        )

        print(
            f"LTP total    : "
            f"{triplet_ltp_ltd['ltp_total']:.10e}"
        )

        print(
            f"LTD total    : "
            f"{triplet_ltp_ltd['ltd_total']:.10e}"
        )

        # ----------------------------------------------------
        # Difference
        # ----------------------------------------------------

        print()
        print("TRIPLET - PAIR")
        print("-" * 40)

        mean_delta_difference = (
            triplet_plasticity["mean"]
            - pair_plasticity["mean"]
        )

        ltp_difference = (
            triplet_ltp_ltd["ltp_mean"]
            - pair_ltp_ltd["ltp_mean"]
        )

        ltd_difference = (
            triplet_ltp_ltd["ltd_mean"]
            - pair_ltp_ltd["ltd_mean"]
        )

        print(
            f"Δ mean difference : "
            f"{mean_delta_difference:+.10e}"
        )

        print(
            f"LTP difference    : "
            f"{ltp_difference:+.10e}"
        )

        print(
            f"LTD difference    : "
            f"{ltd_difference:+.10e}"
        )

        # ----------------------------------------------------
        # Save structured result
        # ----------------------------------------------------

        comparison["layers"][layer] = {
            "pair": {
                "weight_statistics": pair_weight_stats,
                "plasticity_statistics": pair_plasticity,
                "ltp_ltd_statistics": pair_ltp_ltd,
            },
            "triplet": {
                "weight_statistics": triplet_weight_stats,
                "plasticity_statistics": triplet_plasticity,
                "ltp_ltd_statistics": triplet_ltp_ltd,
            },
            "triplet_minus_pair": {
                "mean_delta_difference": float(
                    mean_delta_difference
                ),
                "ltp_difference": float(
                    ltp_difference
                ),
                "ltd_difference": float(
                    ltd_difference
                ),
            },
        }

    # ========================================================
    # Firing rates
    # ========================================================

    pair_rates = extract_firing_rates(pair)
    triplet_rates = extract_firing_rates(triplet)

    print()
    print("=" * 60)
    print("FIRING RATES")
    print("=" * 60)

    if pair_rates is not None:

        print(
            f"Pair Layer 1 : "
            f"{pair_rates['layer1'] * 100:.4f}%"
        )

        print(
            f"Pair Layer 2 : "
            f"{pair_rates['layer2'] * 100:.4f}%"
        )

    else:

        print("Pair firing rates: unavailable")

    print()

    if triplet_rates is not None:

        print(
            f"Triplet Layer 1 : "
            f"{triplet_rates['layer1'] * 100:.4f}%"
        )

        print(
            f"Triplet Layer 2 : "
            f"{triplet_rates['layer2'] * 100:.4f}%"
        )

    else:

        print(
            "Triplet firing rates: unavailable"
        )

    comparison["firing_rates"] = {
        "pair": pair_rates,
        "triplet": triplet_rates,
    }

    # ========================================================
    # Generate plots
    # ========================================================

    print()
    print("=" * 60)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 60)

    plot_mean_weights(
        pair,
        triplet,
        output_dir,
    )

    plot_net_plasticity(
        pair,
        triplet,
        output_dir,
    )

    plot_direction_comparison(
        pair,
        triplet,
        output_dir,
    )

    plot_ltp_ltd(
        pair,
        triplet,
        output_dir,
    )

    plot_firing_rates(
        pair_rates,
        triplet_rates,
        output_dir,
    )

    plot_delta_distributions(
        pair,
        triplet,
        output_dir,
    )

    # ========================================================
    # Save JSON
    # ========================================================

    summary_path = (
        output_dir
        / "comparison_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            comparison,
            f,
            indent=4,
            ensure_ascii=False,
        )

    # ========================================================
    # Final message
    # ========================================================

    print()
    print("=" * 60)
    print("PAIR VS TRIPLET COMPARISON FINISHED")
    print("=" * 60)

    print()
    print("Results saved to:")
    print(output_dir)

    print()
    print("Generated files:")

    for path in sorted(output_dir.iterdir()):

        print(
            f"  {path.name}"
        )

    print()
    print("=" * 60)

if __name__ == "__main__":
    main()