"""
Population-rate control experiment for STDP representation.

Question:

How much class information is contained in the
overall population firing rate alone?

For each sample we calculate:

    population_rate =
        mean firing rate across all neurons and time

Then we evaluate this single scalar feature using:

    1. Nearest centroid
    2. Logistic regression

Comparison:

Full Layer 1 representation      -> 41.32%
Population rate only              -> ???

Full Layer 2 representation      -> 22.65%
Population rate only              -> ???

This experiment helps separate:

    population-level activity
    from
    neuron-specific activity patterns.
"""

import os
import json

import torch
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
# Nearest centroid
# ============================================================

def nearest_centroid_predict(
    X_train,
    y_train,
    X_test,
):
    """
    Predict class using nearest class centroid.
    """

    classes = np.unique(
        y_train
    )

    centroids = []

    for cls in classes:

        centroid = X_train[
            y_train == cls
        ].mean(
            axis=0
        )

        centroids.append(
            centroid
        )

    centroids = np.asarray(
        centroids
    )

    distances = np.linalg.norm(
        X_test[:, None, :]
        -
        centroids[None, :, :],
        axis=2,
    )

    predictions = classes[
        np.argmin(
            distances,
            axis=1,
        )
    ]

    return predictions


# ============================================================
# Evaluate one-dimensional representation
# ============================================================

def evaluate_population_rate(
    X,
    y,
):
    """
    Evaluate a scalar population-rate feature.

    X shape:

        [samples, 1]
    """

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # --------------------------------------------------------
    # Nearest centroid
    # --------------------------------------------------------

    centroid_pred = nearest_centroid_predict(
        X_train_scaled,
        y_train,
        X_test_scaled,
    )

    centroid_accuracy = accuracy_score(
        y_test,
        centroid_pred,
    )

    # --------------------------------------------------------
    # Logistic regression
    # --------------------------------------------------------

    classifier = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )

    classifier.fit(
        X_train_scaled,
        y_train,
    )

    logistic_pred = classifier.predict(
        X_test_scaled
    )

    logistic_accuracy = accuracy_score(
        y_test,
        logistic_pred,
    )

    return {
        "nearest_centroid": float(
            centroid_accuracy
        ),
        "logistic_regression": float(
            logistic_accuracy
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    cfg = Config()

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    stdp_checkpoint = input(
        "\nPath to STDP checkpoint:\n> "
    ).strip()

    if not os.path.exists(
        stdp_checkpoint
    ):

        raise FileNotFoundError(
            f"\nSTDP checkpoint not found:\n"
            f"{stdp_checkpoint}"
        )

    checkpoint_dir = os.path.dirname(
        stdp_checkpoint
    )

    print()
    print("=" * 60)
    print(
        "POPULATION RATE CONTROL EXPERIMENT"
    )
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

    print(
        "\nLoading test dataset..."
    )

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
    # Load STDP checkpoint
    # --------------------------------------------------------

    print(
        "\nLoading STDP weights..."
    )

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

    print(
        "STDP weights loaded."
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    layer1_population_rates = []
    layer2_population_rates = []

    labels_all = []

    # --------------------------------------------------------
    # Extract population rates
    # --------------------------------------------------------

    print(
        "\nExtracting population firing rates..."
    )

    with torch.no_grad():

        for batch_idx, (
            spikes,
            labels,
        ) in enumerate(
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
            # Population firing rate
            #
            # spikes:
            #
            # [time, batch, neurons]
            #
            # mean over time AND neurons
            #
            # -> [batch]
            # ------------------------------------------------

            population_rate1 = (
                spikes1.float()
                .mean(
                    dim=(0, 2)
                )
            )

            population_rate2 = (
                spikes2.float()
                .mean(
                    dim=(0, 2)
                )
            )

            layer1_population_rates.append(
                population_rate1
                .cpu()
                .numpy()
            )

            layer2_population_rates.append(
                population_rate2
                .cpu()
                .numpy()
            )

            labels_all.append(
                labels
                .cpu()
                .numpy()
            )

            if (batch_idx + 1) % 20 == 0:

                print(
                    f"Processed batches: "
                    f"{batch_idx + 1}"
                )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    population_rate1 = np.concatenate(
        layer1_population_rates,
        axis=0,
    )

    population_rate2 = np.concatenate(
        layer2_population_rates,
        axis=0,
    )

    y = np.concatenate(
        labels_all,
        axis=0,
    )

    # Convert [N] -> [N, 1]

    X1 = population_rate1.reshape(
        -1,
        1,
    )

    X2 = population_rate2.reshape(
        -1,
        1,
    )

    print()
    print(
        f"Samples : {len(y)}"
    )

    print(
        f"Layer 1 population feature shape : "
        f"{X1.shape}"
    )

    print(
        f"Layer 2 population feature shape : "
        f"{X2.shape}"
    )

    # ========================================================
    # Basic statistics
    # ========================================================

    print()
    print("=" * 60)
    print(
        "POPULATION RATE STATISTICS"
    )
    print("=" * 60)

    print()

    print(
        f"Layer 1 mean : "
        f"{100 * population_rate1.mean():.4f}%"
    )

    print(
        f"Layer 1 std  : "
        f"{100 * population_rate1.std():.4f}%"
    )

    print()

    print(
        f"Layer 2 mean : "
        f"{100 * population_rate2.mean():.4f}%"
    )

    print(
        f"Layer 2 std  : "
        f"{100 * population_rate2.std():.4f}%"
    )

    # ========================================================
    # Class means
    # ========================================================

    print()
    print("=" * 60)
    print(
        "CLASS POPULATION RATES"
    )
    print("=" * 60)

    class_means_l1 = []
    class_means_l2 = []

    for cls in range(
        cfg.n_classes
    ):

        mask = (
            y == cls
        )

        mean_l1 = population_rate1[
            mask
        ].mean()

        mean_l2 = population_rate2[
            mask
        ].mean()

        class_means_l1.append(
            float(mean_l1)
        )

        class_means_l2.append(
            float(mean_l2)
        )

        print(
            f"{CLASS_NAMES[cls]:>5} : "
            f"Layer1 = "
            f"{100 * mean_l1:7.3f}%   "
            f"Layer2 = "
            f"{100 * mean_l2:7.3f}%"
        )

    # ========================================================
    # Evaluate
    # ========================================================

    print()
    print("=" * 60)
    print(
        "POPULATION RATE CLASSIFICATION"
    )
    print("=" * 60)

    print(
        "\nLayer 1:"
    )

    result_l1 = evaluate_population_rate(
        X1,
        y,
    )

    print(
        f"Nearest centroid : "
        f"{100 * result_l1['nearest_centroid']:.2f}%"
    )

    print(
        f"Logistic regression : "
        f"{100 * result_l1['logistic_regression']:.2f}%"
    )

    print(
        "\nLayer 2:"
    )

    result_l2 = evaluate_population_rate(
        X2,
        y,
    )

    print(
        f"Nearest centroid : "
        f"{100 * result_l2['nearest_centroid']:.2f}%"
    )

    print(
        f"Logistic regression : "
        f"{100 * result_l2['logistic_regression']:.2f}%"
    )

    # ========================================================
    # Comparison with full representation
    # ========================================================

    print()
    print("=" * 60)
    print(
        "COMPARISON WITH FULL REPRESENTATION"
    )
    print("=" * 60)

    # Previous results

    full_l1_logistic = 0.4132
    full_l2_logistic = 0.2265

    full_l1_centroid = 0.2191
    full_l2_centroid = 0.2324

    print()
    print(
        f"{'Metric':<30}"
        f"{'Full':>12}"
        f"{'Population':>15}"
    )

    print("-" * 57)

    print(
        f"{'Layer 1 — centroid':<30}"
        f"{100 * full_l1_centroid:>11.2f}%"
        f"{100 * result_l1['nearest_centroid']:>14.2f}%"
    )

    print(
        f"{'Layer 1 — logistic':<30}"
        f"{100 * full_l1_logistic:>11.2f}%"
        f"{100 * result_l1['logistic_regression']:>14.2f}%"
    )

    print(
        f"{'Layer 2 — centroid':<30}"
        f"{100 * full_l2_centroid:>11.2f}%"
        f"{100 * result_l2['nearest_centroid']:>14.2f}%"
    )

    print(
        f"{'Layer 2 — logistic':<30}"
        f"{100 * full_l2_logistic:>11.2f}%"
        f"{100 * result_l2['logistic_regression']:>14.2f}%"
    )

    # ========================================================
    # Save results
    # ========================================================

    results = {

        "checkpoint": stdp_checkpoint,

        "n_samples": int(
            len(y)
        ),

        "layer1": {

            "population_rate_mean":
                float(
                    population_rate1.mean()
                ),

            "population_rate_std":
                float(
                    population_rate1.std()
                ),

            "class_means":
                class_means_l1,

            "nearest_centroid":
                result_l1[
                    "nearest_centroid"
                ],

            "logistic_regression":
                result_l1[
                    "logistic_regression"
                ],

            "full_representation_nearest_centroid":
                full_l1_centroid,

            "full_representation_logistic":
                full_l1_logistic,
        },

        "layer2": {

            "population_rate_mean":
                float(
                    population_rate2.mean()
                ),

            "population_rate_std":
                float(
                    population_rate2.std()
                ),

            "class_means":
                class_means_l2,

            "nearest_centroid":
                result_l2[
                    "nearest_centroid"
                ],

            "logistic_regression":
                result_l2[
                    "logistic_regression"
                ],

            "full_representation_nearest_centroid":
                full_l2_centroid,

            "full_representation_logistic":
                full_l2_logistic,
        },
    }

    json_path = os.path.join(
        checkpoint_dir,
        "population_rate_control.json",
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
    print(
        "Population-rate control finished."
    )
    print("=" * 60)

    print(
        f"\nResults saved to:"
    )

    print(
        json_path
    )


if __name__ == "__main__":
    main()