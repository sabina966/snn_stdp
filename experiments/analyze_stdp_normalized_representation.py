"""
Control experiment for STDP representation.

Tests whether classification performance is mainly caused by
the overall firing intensity of each sample.

For every sample we compare:

1. Raw representation:
       [r1, r2, ..., rN]

2. Normalized representation:
       [r1 / sum(r), ..., rN / sum(r)]

Pipeline:

STDP checkpoint
        |
        v
SHD test set
        |
        v
HierarchicalSNN
        |
        +--> Layer 1 firing-rate representation
        |
        +--> Layer 2 firing-rate representation
        |
        v
Raw vs normalized representation
        |
        +--> Logistic regression
        +--> Nearest centroid
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

    classes = np.unique(y_train)

    centroids = []

    for cls in classes:

        centroid = X_train[
            y_train == cls
        ].mean(axis=0)

        centroids.append(
            centroid
        )

    centroids = np.asarray(
        centroids
    )

    distances = np.linalg.norm(
        X_test[:, None, :] -
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
# Evaluate representation
# ============================================================

def evaluate_representation(
    X,
    y,
):
    """
    Evaluate a representation using:

    1. Nearest centroid
    2. Logistic regression
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
# Normalize representation
# ============================================================

def normalize_representation(
    X,
    eps=1e-12,
):
    """
    Remove the overall firing-rate magnitude.

    Each sample is normalized by its total activity:

        x_i' = x_i / sum(x)

    Therefore the classifier mainly sees the
    relative distribution of activity between neurons.
    """

    total_activity = X.sum(
        axis=1,
        keepdims=True,
    )

    X_normalized = X / (
        total_activity + eps
    )

    return X_normalized


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
        "STDP NORMALIZATION CONTROL EXPERIMENT"
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
    # Load checkpoint
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

    layer1_representations = []
    layer2_representations = []
    labels_all = []

    # --------------------------------------------------------
    # Extract representations
    # --------------------------------------------------------

    print(
        "\nExtracting representations..."
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
            # Keep individual neurons as features
            #
            # [time, batch, neurons]
            #       ↓ mean over time
            # [batch, neurons]
            # ------------------------------------------------

            rate1 = spikes1.float().mean(
                dim=0
            )

            rate2 = spikes2.float().mean(
                dim=0
            )

            layer1_representations.append(
                rate1.cpu().numpy()
            )

            layer2_representations.append(
                rate2.cpu().numpy()
            )

            labels_all.append(
                labels.cpu().numpy()
            )

            if (batch_idx + 1) % 20 == 0:

                print(
                    f"Processed batches: "
                    f"{batch_idx + 1}"
                )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    X1 = np.concatenate(
        layer1_representations,
        axis=0,
    )

    X2 = np.concatenate(
        layer2_representations,
        axis=0,
    )

    y = np.concatenate(
        labels_all,
        axis=0,
    )

    print()
    print(
        f"Samples       : {len(y)}"
    )

    print(
        f"Layer 1 shape : {X1.shape}"
    )

    print(
        f"Layer 2 shape : {X2.shape}"
    )

    # ========================================================
    # Raw representations
    # ========================================================

    print()
    print("=" * 60)
    print("RAW REPRESENTATIONS")
    print("=" * 60)

    raw_l1 = evaluate_representation(
        X1,
        y,
    )

    raw_l2 = evaluate_representation(
        X2,
        y,
    )

    print(
        "\nLayer 1"
    )

    print(
        f"Nearest centroid : "
        f"{100 * raw_l1['nearest_centroid']:.2f}%"
    )

    print(
        f"Logistic regression : "
        f"{100 * raw_l1['logistic_regression']:.2f}%"
    )

    print(
        "\nLayer 2"
    )

    print(
        f"Nearest centroid : "
        f"{100 * raw_l2['nearest_centroid']:.2f}%"
    )

    print(
        f"Logistic regression : "
        f"{100 * raw_l2['logistic_regression']:.2f}%"
    )

    # ========================================================
    # Normalized representations
    # ========================================================

    print()
    print("=" * 60)
    print("NORMALIZED REPRESENTATIONS")
    print("=" * 60)

    print(
        "\nRemoving overall firing-rate magnitude..."
    )

    X1_normalized = normalize_representation(
        X1
    )

    X2_normalized = normalize_representation(
        X2
    )

    normalized_l1 = evaluate_representation(
        X1_normalized,
        y,
    )

    normalized_l2 = evaluate_representation(
        X2_normalized,
        y,
    )

    print(
        "\nLayer 1"
    )

    print(
        f"Nearest centroid : "
        f"{100 * normalized_l1['nearest_centroid']:.2f}%"
    )

    print(
        f"Logistic regression : "
        f"{100 * normalized_l1['logistic_regression']:.2f}%"
    )

    print(
        "\nLayer 2"
    )

    print(
        f"Nearest centroid : "
        f"{100 * normalized_l2['nearest_centroid']:.2f}%"
    )

    print(
        f"Logistic regression : "
        f"{100 * normalized_l2['logistic_regression']:.2f}%"
    )

    # ========================================================
    # Comparison
    # ========================================================

    print()
    print("=" * 60)
    print("RAW vs NORMALIZED")
    print("=" * 60)

    print()
    print(
        f"{'Metric':<30}"
        f"{'Raw':>12}"
        f"{'Normalized':>15}"
    )

    print("-" * 57)

    print(
        f"{'Layer 1 — centroid':<30}"
        f"{100 * raw_l1['nearest_centroid']:>11.2f}%"
        f"{100 * normalized_l1['nearest_centroid']:>14.2f}%"
    )

    print(
        f"{'Layer 1 — logistic':<30}"
        f"{100 * raw_l1['logistic_regression']:>11.2f}%"
        f"{100 * normalized_l1['logistic_regression']:>14.2f}%"
    )

    print(
        f"{'Layer 2 — centroid':<30}"
        f"{100 * raw_l2['nearest_centroid']:>11.2f}%"
        f"{100 * normalized_l2['nearest_centroid']:>14.2f}%"
    )

    print(
        f"{'Layer 2 — logistic':<30}"
        f"{100 * raw_l2['logistic_regression']:>11.2f}%"
        f"{100 * normalized_l2['logistic_regression']:>14.2f}%"
    )

    # ========================================================
    # Differences
    # ========================================================

    l1_logistic_difference = (
        normalized_l1[
            "logistic_regression"
        ]
        -
        raw_l1[
            "logistic_regression"
        ]
    )

    l2_logistic_difference = (
        normalized_l2[
            "logistic_regression"
        ]
        -
        raw_l2[
            "logistic_regression"
        ]
    )

    print()
    print(
        "Change in logistic-regression accuracy:"
    )

    print(
        f"Layer 1 : "
        f"{100 * l1_logistic_difference:+.2f} percentage points"
    )

    print(
        f"Layer 2 : "
        f"{100 * l2_logistic_difference:+.2f} percentage points"
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

            "raw": raw_l1,

            "normalized": normalized_l1,

            "logistic_difference": float(
                l1_logistic_difference
            ),
        },

        "layer2": {

            "raw": raw_l2,

            "normalized": normalized_l2,

            "logistic_difference": float(
                l2_logistic_difference
            ),
        },
    }

    json_path = os.path.join(
        checkpoint_dir,
        "normalization_control.json",
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
    print("Control experiment finished.")
    print("=" * 60)

    print(
        f"\nResults saved to:"
    )

    print(
        json_path
    )


if __name__ == "__main__":
    main()