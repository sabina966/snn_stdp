"""
Analyze neuron-specific activity of a STDP-pretrained SNN.

Pipeline:

STDP checkpoint
        |
        v
SHD test set
        |
        v
HierarchicalSNN
        |
        +--> Layer 1 neuron activity
        |
        +--> Layer 2 neuron activity
        |
        v
Class x neuron firing-rate heatmaps
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

    checkpoint_dir = os.path.dirname(
        stdp_checkpoint
    )

    print()
    print("=" * 60)
    print("STDP neuron-specific activity analysis")
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
    # Number of neurons
    # --------------------------------------------------------

    n_hidden1 = cfg.hidden1
    n_hidden2 = cfg.hidden2
    n_classes = cfg.n_classes

    print()
    print(
        f"Layer 1 neurons : {n_hidden1}"
    )

    print(
        f"Layer 2 neurons : {n_hidden2}"
    )

    # --------------------------------------------------------
    # Storage
    #
    # class x neuron
    # --------------------------------------------------------

    layer1_activity = np.zeros(
        (n_classes, n_hidden1),
        dtype=np.float64,
    )

    layer2_activity = np.zeros(
        (n_classes, n_hidden2),
        dtype=np.float64,
    )

    class_count = np.zeros(
        n_classes,
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Test set
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
            # Forward
            # ------------------------------------------------

            _, spikes1, spikes2 = model(
                spikes,
                apply_stdp=False,
                return_activity=True,
            )

            # ------------------------------------------------
            # Average activity over time
            #
            # spikes1:
            # [time, batch, hidden1]
            #
            # spikes2:
            # [time, batch, hidden2]
            #
            # result:
            # [batch, neurons]
            # ------------------------------------------------

            activity1 = spikes1.float().mean(
                dim=0
            )

            activity2 = spikes2.float().mean(
                dim=0
            )

            labels_cpu = labels.cpu().numpy()

            activity1_cpu = (
                activity1.cpu().numpy()
            )

            activity2_cpu = (
                activity2.cpu().numpy()
            )

            # ------------------------------------------------
            # Accumulate activity by class
            # ------------------------------------------------

            for i, label in enumerate(
                labels_cpu
            ):

                label = int(label)

                layer1_activity[label] += (
                    activity1_cpu[i]
                )

                layer2_activity[label] += (
                    activity2_cpu[i]
                )

                class_count[label] += 1

            if (batch_idx + 1) % 20 == 0:

                print(
                    f"Processed batches: "
                    f"{batch_idx + 1}"
                )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    for class_idx in range(n_classes):

        if class_count[class_idx] > 0:

            layer1_activity[class_idx] /= (
                class_count[class_idx]
            )

            layer2_activity[class_idx] /= (
                class_count[class_idx]
            )

    # --------------------------------------------------------
    # Convert to percentage
    # --------------------------------------------------------

    layer1_percent = (
        100 * layer1_activity
    )

    layer2_percent = (
        100 * layer2_activity
    )

    # --------------------------------------------------------
    # Print basic statistics
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("NEURON-SPECIFIC ACTIVITY")
    print("=" * 60)

    print(
        f"Layer 1:"
    )

    print(
        f"  Minimum activity : "
        f"{layer1_percent.min():.4f}%"
    )

    print(
        f"  Maximum activity : "
        f"{layer1_percent.max():.4f}%"
    )

    print(
        f"  Mean activity    : "
        f"{layer1_percent.mean():.4f}%"
    )

    print()

    print(
        f"Layer 2:"
    )

    print(
        f"  Minimum activity : "
        f"{layer2_percent.min():.4f}%"
    )

    print(
        f"  Maximum activity : "
        f"{layer2_percent.max():.4f}%"
    )

    print(
        f"  Mean activity    : "
        f"{layer2_percent.mean():.4f}%"
    )

    # ========================================================
    # Most class-selective neurons
    # ========================================================

    print()
    print("=" * 60)
    print("MOST CLASS-SELECTIVE NEURONS")
    print("=" * 60)

    # --------------------------------------------------------
    # Layer 1
    # --------------------------------------------------------

    layer1_selectivity = (
        layer1_percent.max(axis=0)
        -
        layer1_percent.min(axis=0)
    )

    top1 = np.argsort(
        layer1_selectivity
    )[::-1][:10]

    print("\nLayer 1:")

    for neuron in top1:

        preferred_class = np.argmax(
            layer1_percent[:, neuron]
        )

        print(
            f"Neuron {neuron:3d} | "
            f"selectivity = "
            f"{layer1_selectivity[neuron]:7.3f}% | "
            f"preferred = "
            f"{CLASS_NAMES[preferred_class]}"
        )

    # --------------------------------------------------------
    # Layer 2
    # --------------------------------------------------------

    layer2_selectivity = (
        layer2_percent.max(axis=0)
        -
        layer2_percent.min(axis=0)
    )

    top2 = np.argsort(
        layer2_selectivity
    )[::-1][:10]

    print("\nLayer 2:")

    for neuron in top2:

        preferred_class = np.argmax(
            layer2_percent[:, neuron]
        )

        print(
            f"Neuron {neuron:3d} | "
            f"selectivity = "
            f"{layer2_selectivity[neuron]:7.3f}% | "
            f"preferred = "
            f"{CLASS_NAMES[preferred_class]}"
        )

    # ========================================================
    # Heatmap Layer 1
    # ========================================================

    plt.figure(
        figsize=(16, 8)
    )

    plt.imshow(
        layer1_percent,
        aspect="auto",
    )

    plt.colorbar(
        label="Firing rate (%)"
    )

    plt.xticks(
        range(n_hidden1),
        range(n_hidden1),
        fontsize=7,
    )

    plt.yticks(
        range(n_classes),
        CLASS_NAMES,
    )

    plt.xlabel(
        "Нейрон Layer 1"
    )

    plt.ylabel(
        "Клас SHD"
    )

    plt.title(
        "Актывацыя нейронаў Layer 1 для розных класаў"
    )

    plt.tight_layout()

    layer1_path = os.path.join(
        checkpoint_dir,
        "layer1_class_neuron_activity.png",
    )

    plt.savefig(
        layer1_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # Heatmap Layer 2
    # ========================================================

    plt.figure(
        figsize=(14, 8)
    )

    plt.imshow(
        layer2_percent,
        aspect="auto",
    )

    plt.colorbar(
        label="Firing rate (%)"
    )

    plt.xticks(
        range(n_hidden2),
        range(n_hidden2),
    )

    plt.yticks(
        range(n_classes),
        CLASS_NAMES,
    )

    plt.xlabel(
        "Нейрон Layer 2"
    )

    plt.ylabel(
        "Клас SHD"
    )

    plt.title(
        "Актывацыя нейронаў Layer 2 для розных класаў"
    )

    plt.tight_layout()

    layer2_path = os.path.join(
        checkpoint_dir,
        "layer2_class_neuron_activity.png",
    )

    plt.savefig(
        layer2_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # Save JSON
    # ========================================================

    results = {

        "class_names":
            CLASS_NAMES,

        "class_count":
            class_count.tolist(),

        "layer1_activity":
            layer1_activity.tolist(),

        "layer2_activity":
            layer2_activity.tolist(),

        "layer1_selectivity":
            layer1_selectivity.tolist(),

        "layer2_selectivity":
            layer2_selectivity.tolist(),

        "top_layer1_neurons":
            [
                {
                    "neuron": int(n),
                    "selectivity": float(
                        layer1_selectivity[n]
                    ),
                    "preferred_class":
                        CLASS_NAMES[
                            int(
                                np.argmax(
                                    layer1_percent[:, n]
                                )
                            )
                        ],
                }
                for n in top1
            ],

        "top_layer2_neurons":
            [
                {
                    "neuron": int(n),
                    "selectivity": float(
                        layer2_selectivity[n]
                    ),
                    "preferred_class":
                        CLASS_NAMES[
                            int(
                                np.argmax(
                                    layer2_percent[:, n]
                                )
                            )
                        ],
                }
                for n in top2
            ],
    }

    json_path = os.path.join(
        checkpoint_dir,
        "neuron_activity.json",
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

    # ========================================================
    # Finished
    # ========================================================

    print()
    print("=" * 60)
    print("Analysis finished.")
    print("=" * 60)

    print(
        f"\nLayer 1 heatmap:\n"
        f"{layer1_path}"
    )

    print(
        f"\nLayer 2 heatmap:\n"
        f"{layer2_path}"
    )

    print(
        f"\nJSON:\n"
        f"{json_path}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()