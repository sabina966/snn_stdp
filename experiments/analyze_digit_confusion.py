"""
Final digit-level analysis of STDP representations.

SHD classes:

    EN_0 ... EN_9
    DE_0 ... DE_9

Here English and German versions of the same digit
are merged:

    EN_0 + DE_0 -> digit 0
    ...
    EN_9 + DE_9 -> digit 9

For Layer 1 and Layer 2:

    1. Train logistic regression on the STDP representation
    2. Calculate digit confusion matrix
    3. Calculate per-digit accuracy
    4. Compare Layer 1 and Layer 2

This is the final detailed digit-level analysis.
"""

import os
import json

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
)


# ============================================================
# Main
# ============================================================

def evaluate_digit_representation(
    X,
    y_digit,
):
    """
    Train logistic regression and return:

        accuracy
        predictions
        true labels
        confusion matrix
        per-class accuracy
    """

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y_digit,
        test_size=0.30,
        random_state=42,
        stratify=y_digit,
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

    predictions = classifier.predict(
        X_test_scaled
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(10),
    )

    # --------------------------------------------------------
    # Per-digit accuracy
    # --------------------------------------------------------

    per_digit_accuracy = []

    for digit in range(10):

        mask = (
            y_test == digit
        )

        if np.sum(mask) == 0:

            acc = 0.0

        else:

            acc = np.mean(
                predictions[mask] == digit
            )

        per_digit_accuracy.append(
            float(acc)
        )

    return {
        "accuracy": float(accuracy),
        "confusion_matrix": cm,
        "per_digit_accuracy":
            per_digit_accuracy,
        "y_test": y_test,
        "predictions": predictions,
    }


# ============================================================
# Plot confusion matrix
# ============================================================

def plot_confusion_matrix(
    cm,
    accuracy,
    layer_name,
    output_path,
):
    """
    Plot normalized confusion matrix.
    """

    row_sums = cm.sum(
        axis=1,
        keepdims=True,
    )

    normalized_cm = (
        cm /
        np.maximum(
            row_sums,
            1,
        )
    )

    plt.figure(
        figsize=(9, 8)
    )

    plt.imshow(
        normalized_cm,
        interpolation="nearest",
        aspect="auto",
    )

    plt.colorbar(
        label="Доля предсказаний"
    )

    plt.xticks(
        np.arange(10),
        [str(i) for i in range(10)],
    )

    plt.yticks(
        np.arange(10),
        [str(i) for i in range(10)],
    )

    plt.xlabel(
        "Предсказанная цифра"
    )

    plt.ylabel(
        "Истинная цифра"
    )

    plt.title(
        f"{layer_name}: digit confusion matrix\n"
        f"Logistic regression accuracy = "
        f"{100 * accuracy:.2f}%"
    )

    # --------------------------------------------------------
    # Add percentages
    # --------------------------------------------------------

    for i in range(10):

        for j in range(10):

            value = (
                100 *
                normalized_cm[i, j]
            )

            plt.text(
                j,
                i,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Plot per-digit accuracy
# ============================================================

def plot_per_digit_accuracy(
    layer1_accuracy,
    layer2_accuracy,
    output_path,
):
    """
    Compare per-digit accuracy for both layers.
    """

    digits = np.arange(10)

    width = 0.35

    plt.figure(
        figsize=(12, 7)
    )

    plt.bar(
        digits - width / 2,
        100 * np.asarray(
            layer1_accuracy
        ),
        width,
        label="Layer 1",
    )

    plt.bar(
        digits + width / 2,
        100 * np.asarray(
            layer2_accuracy
        ),
        width,
        label="Layer 2",
    )

    plt.xticks(
        digits,
        [str(i) for i in digits],
    )

    plt.xlabel(
        "Цыфра"
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.title(
        "Per-digit classification accuracy"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

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

    # --------------------------------------------------------
    # Representation file
    # --------------------------------------------------------

    representation_path = os.path.join(
        checkpoint_dir,
        "representation_analysis.json",
    )

    if not os.path.exists(
        representation_path
    ):

        raise FileNotFoundError(
            "\nrepresentation_analysis.json "
            "not found.\n"
            f"Expected:\n"
            f"{representation_path}"
        )

    print()
    print("=" * 60)
    print(
        "FINAL DIGIT-LEVEL STDP ANALYSIS"
    )
    print("=" * 60)

    print(
        f"\nCheckpoint : "
        f"{stdp_checkpoint}"
    )

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    #
    # We need the actual representations.
    #
    # The JSON contains metrics but not the
    # 200/100 dimensional sample representations.
    #
    # Therefore we reproduce the representation
    # extraction here.
    # --------------------------------------------------------

    from configs import Config
    from datasets.dataloader import get_dataloaders
    from network.hierarchical_snn import HierarchicalSNN

    cfg = Config()

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
    # Extract representations
    # --------------------------------------------------------

    layer1_features = []
    layer2_features = []
    labels_all = []

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

            _, spikes1, spikes2 = model(
                spikes,
                apply_stdp=False,
                return_activity=True,
            )

            # [time, batch, neurons]
            #
            # ->
            #
            # [batch, neurons]

            rate1 = spikes1.float().mean(
                dim=0
            )

            rate2 = spikes2.float().mean(
                dim=0
            )

            layer1_features.append(
                rate1.cpu().numpy()
            )

            layer2_features.append(
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
        layer1_features,
        axis=0,
    )

    X2 = np.concatenate(
        layer2_features,
        axis=0,
    )

    y20 = np.concatenate(
        labels_all,
        axis=0,
    )

    # --------------------------------------------------------
    # Convert 20 classes -> 10 digits
    # --------------------------------------------------------

    y_digit = (
        y20 % 10
    )

    print()
    print(
        f"Samples       : {len(y_digit)}"
    )

    print(
        f"Layer 1 shape : {X1.shape}"
    )

    print(
        f"Layer 2 shape : {X2.shape}"
    )

    # ========================================================
    # Evaluate Layer 1
    # ========================================================

    print()
    print("=" * 60)
    print(
        "LAYER 1"
    )
    print("=" * 60)

    result_l1 = evaluate_digit_representation(
        X1,
        y_digit,
    )

    print()
    print(
        f"Overall digit accuracy : "
        f"{100 * result_l1['accuracy']:.2f}%"
    )

    print()
    print(
        "Per-digit accuracy:"
    )

    for digit, acc in enumerate(
        result_l1[
            "per_digit_accuracy"
        ]
    ):

        print(
            f"Digit {digit} : "
            f"{100 * acc:6.2f}%"
        )

    # ========================================================
    # Evaluate Layer 2
    # ========================================================

    print()
    print("=" * 60)
    print(
        "LAYER 2"
    )
    print("=" * 60)

    result_l2 = evaluate_digit_representation(
        X2,
        y_digit,
    )

    print()
    print(
        f"Overall digit accuracy : "
        f"{100 * result_l2['accuracy']:.2f}%"
    )

    print()
    print(
        "Per-digit accuracy:"
    )

    for digit, acc in enumerate(
        result_l2[
            "per_digit_accuracy"
        ]
    ):

        print(
            f"Digit {digit} : "
            f"{100 * acc:6.2f}%"
        )

    # ========================================================
    # Confusion matrices
    # ========================================================

    print()
    print(
        "Creating confusion matrices..."
    )

    cm1_path = os.path.join(
        checkpoint_dir,
        "digit_confusion_layer1.png",
    )

    cm2_path = os.path.join(
        checkpoint_dir,
        "digit_confusion_layer2.png",
    )

    plot_confusion_matrix(
        result_l1[
            "confusion_matrix"
        ],
        result_l1[
            "accuracy"
        ],
        "Layer 1",
        cm1_path,
    )

    plot_confusion_matrix(
        result_l2[
            "confusion_matrix"
        ],
        result_l2[
            "accuracy"
        ],
        "Layer 2",
        cm2_path,
    )

    # ========================================================
    # Per-digit plot
    # ========================================================

    per_digit_path = os.path.join(
        checkpoint_dir,
        "digit_per_class_accuracy.png",
    )

    plot_per_digit_accuracy(
        result_l1[
            "per_digit_accuracy"
        ],
        result_l2[
            "per_digit_accuracy"
        ],
        per_digit_path,
    )

    # ========================================================
    # Print confusion matrices
    # ========================================================

    print()
    print("=" * 60)
    print(
        "LAYER 1 CONFUSION MATRIX"
    )
    print("=" * 60)

    print(
        result_l1[
            "confusion_matrix"
        ]
    )

    print()
    print("=" * 60)
    print(
        "LAYER 2 CONFUSION MATRIX"
    )
    print("=" * 60)

    print(
        result_l2[
            "confusion_matrix"
        ]
    )

    # ========================================================
    # Save JSON
    # ========================================================

    results = {

        "checkpoint":
            stdp_checkpoint,

        "n_samples":
            int(len(y_digit)),

        "layer1": {

            "accuracy":
                result_l1[
                    "accuracy"
                ],

            "per_digit_accuracy":
                result_l1[
                    "per_digit_accuracy"
                ],

            "confusion_matrix":
                result_l1[
                    "confusion_matrix"
                ].tolist(),
        },

        "layer2": {

            "accuracy":
                result_l2[
                    "accuracy"
                ],

            "per_digit_accuracy":
                result_l2[
                    "per_digit_accuracy"
                ],

            "confusion_matrix":
                result_l2[
                    "confusion_matrix"
                ].tolist(),
        },
    }

    json_path = os.path.join(
        checkpoint_dir,
        "digit_confusion_analysis.json",
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
        "FINAL DIGIT ANALYSIS FINISHED"
    )
    print("=" * 60)

    print(
        "\nSaved:"
    )

    print(
        json_path
    )

    print(
        cm1_path
    )

    print(
        cm2_path
    )

    print(
        per_digit_path
    )


if __name__ == "__main__":
    main()