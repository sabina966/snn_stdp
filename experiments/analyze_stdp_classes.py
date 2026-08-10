"""
Analyze class-dependent activity of a STDP-pretrained SNN.

Pipeline:

STDP checkpoint
        |
        v
SHD test set
        |
        v
HierarchicalSNN
        |
        +--> Layer 1 firing rate
        |
        +--> Layer 2 firing rate
        |
        v
Class-dependent activity analysis
"""

import os
import json

import torch
import numpy as np
import matplotlib.pyplot as plt

from configs import Config
from datasets.dataloader import get_dataloaders
from network.hierarchical_snn import HierarchicalSNN


# ============================================================
# SHD class names
# ============================================================

CLASS_NAMES = [
    "EN_0",
    "EN_1",
    "EN_2",
    "EN_3",
    "EN_4",
    "EN_5",
    "EN_6",
    "EN_7",
    "EN_8",
    "EN_9",
    "DE_0",
    "DE_1",
    "DE_2",
    "DE_3",
    "DE_4",
    "DE_5",
    "DE_6",
    "DE_7",
    "DE_8",
    "DE_9",
]


# ============================================================
# Main
# ============================================================

def main():

    cfg = Config()

    # --------------------------------------------------------
    # STDP checkpoint
    # --------------------------------------------------------

    stdp_checkpoint = input(
        "\nPath to STDP checkpoint:\n> "
    ).strip()

    if not os.path.exists(stdp_checkpoint):

        raise FileNotFoundError(
            f"\nSTDP checkpoint not found:\n"
            f"{stdp_checkpoint}"
        )

    # --------------------------------------------------------
    # Results directory
    # --------------------------------------------------------

    checkpoint_dir = os.path.dirname(
        stdp_checkpoint
    )

    print()
    print("=" * 60)
    print("STDP class-dependent activity analysis")
    print("=" * 60)

    print(
        f"Checkpoint : {stdp_checkpoint}"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        cfg.device
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device     : {device}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nLoading test dataset...")

    _, test_loader = get_dataloaders(
        root=cfg.dataset_root,
        batch_size=cfg.batch_size,
        time_steps=cfg.time_steps,
    )

    print("Done.")

    # --------------------------------------------------------
    # Network parameters
    # --------------------------------------------------------

    neuron_params = dict(
        tau_m=cfg.tau_m,
        v_rest=cfg.v_rest,
        v_reset=cfg.v_reset,
        v_threshold=cfg.v_threshold,
        tau_adaptation=cfg.tau_adaptation,
        adaptation_strength=cfg.adaptation_strength,
    )

    stdp_params = dict(
        a_plus=cfg.a_plus,
        a_minus=cfg.a_minus,
        tau_plus=cfg.tau_plus,
        tau_minus=cfg.tau_minus,
        w_min=cfg.w_min,
        w_max=cfg.w_max,
    )

    homeostasis_params = dict(
        target_rate=cfg.target_rate,
        tau_homeostasis=cfg.tau_homeostasis,
        strength=cfg.homeostasis_strength,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = HierarchicalSNN(
        n_input=cfg.n_input,
        n_hidden1=cfg.hidden1,
        n_hidden2=cfg.hidden2,
        n_classes=cfg.n_classes,

        neuron_params=neuron_params,

        stdp_params1=stdp_params,
        stdp_params2=stdp_params,

        homeostasis_params=homeostasis_params,

        input_gain=cfg.input_gain,
    ).to(device)

    # --------------------------------------------------------
    # Load STDP weights
    # --------------------------------------------------------

    print("\nLoading STDP weights...")

    checkpoint = torch.load(
        stdp_checkpoint,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint,
        strict=True,
    )

    model.eval()

    print("STDP weights loaded.")

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    n_classes = cfg.n_classes

    layer1_rate_sum = np.zeros(
        n_classes,
        dtype=np.float64,
    )

    layer2_rate_sum = np.zeros(
        n_classes,
        dtype=np.float64,
    )

    class_count = np.zeros(
        n_classes,
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Run test set
    # --------------------------------------------------------

    print("\nRunning test set...\n")

    with torch.no_grad():

        for batch_idx, (spikes, labels) in enumerate(
            test_loader
        ):

            spikes = spikes.permute(
                1,
                0,
                2,
            ).to(device)

            labels = labels.to(device)

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            _, spikes1, spikes2 = model(
                spikes,
                apply_stdp=False,
                return_activity=True,
            )

            # ------------------------------------------------
            # Firing rate per sample
            #
            # spikes shape:
            # [time, batch, neurons]
            #
            # mean over time and neurons
            # -> [batch]
            # ------------------------------------------------

            rate1 = spikes1.float().mean(
                dim=(0, 2)
            )

            rate2 = spikes2.float().mean(
                dim=(0, 2)
            )

            # ------------------------------------------------
            # Accumulate by class
            # ------------------------------------------------

            labels_cpu = labels.cpu().numpy()
            rate1_cpu = rate1.cpu().numpy()
            rate2_cpu = rate2.cpu().numpy()

            for i, label in enumerate(
                labels_cpu
            ):

                label = int(label)

                layer1_rate_sum[label] += (
                    rate1_cpu[i]
                )

                layer2_rate_sum[label] += (
                    rate2_cpu[i]
                )

                class_count[label] += 1

            if (batch_idx + 1) % 20 == 0:

                print(
                    f"Processed batches: "
                    f"{batch_idx + 1}"
                )

    # --------------------------------------------------------
    # Calculate means
    # --------------------------------------------------------

    layer1_rates = (
        layer1_rate_sum /
        np.maximum(class_count, 1)
    )

    layer2_rates = (
        layer2_rate_sum /
        np.maximum(class_count, 1)
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("CLASS-DEPENDENT FIRING RATES")
    print("=" * 60)

    for i in range(n_classes):

        print(
            f"{CLASS_NAMES[i]:>5} : "
            f"Layer1 = "
            f"{100 * layer1_rates[i]:7.3f}%   "
            f"Layer2 = "
            f"{100 * layer2_rates[i]:7.3f}%   "
            f"N = {class_count[i]}"
        )

    # --------------------------------------------------------
    # Language averages
    # --------------------------------------------------------

    english_layer1 = layer1_rates[:10].mean()
    english_layer2 = layer2_rates[:10].mean()

    german_layer1 = layer1_rates[10:].mean()
    german_layer2 = layer2_rates[10:].mean()

    print()
    print("=" * 60)
    print("LANGUAGE AVERAGES")
    print("=" * 60)

    print(
        f"English : "
        f"Layer1 = {100 * english_layer1:.3f}%   "
        f"Layer2 = {100 * english_layer2:.3f}%"
    )

    print(
        f"German  : "
        f"Layer1 = {100 * german_layer1:.3f}%   "
        f"Layer2 = {100 * german_layer2:.3f}%"
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    results = {

        "class_names": CLASS_NAMES,

        "class_count":
            class_count.tolist(),

        "layer1_rate":
            layer1_rates.tolist(),

        "layer2_rate":
            layer2_rates.tolist(),

        "english": {
            "layer1_rate":
                float(english_layer1),

            "layer2_rate":
                float(english_layer2),
        },

        "german": {
            "layer1_rate":
                float(german_layer1),

            "layer2_rate":
                float(german_layer2),
        },
    }

    json_path = os.path.join(
        checkpoint_dir,
        "class_activity.json",
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    x = np.arange(n_classes)

    width = 0.35

    plt.figure(
        figsize=(14, 7)
    )

    plt.bar(
        x - width / 2,
        100 * layer1_rates,
        width,
        label="Layer 1",
    )

    plt.bar(
        x + width / 2,
        100 * layer2_rates,
        width,
        label="Layer 2",
    )

    plt.xticks(
        x,
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )

    plt.xlabel(
        "Клас SHD"
    )

    plt.ylabel(
        "Сярэдняя частата спайкаў (%)"
    )

    plt.title(
        "Залежнасць спайкавай актыўнасці ад класа"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plot_path = os.path.join(
        checkpoint_dir,
        "class_activity.png",
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Analysis finished.")
    print("=" * 60)

    print(
        f"\nSaved to:\n"
        f"{json_path}"
    )

    print(
        f"\nPlot saved to:\n"
        f"{plot_path}"
    )


if __name__ == "__main__":
    main()