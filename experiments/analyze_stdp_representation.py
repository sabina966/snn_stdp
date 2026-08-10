"""
Analyze the representation formed by STDP-pretraining.

Pipeline:

STDP checkpoint
        |
        v
SHD test set
        |
        v
HierarchicalSNN
        |
        +--> Layer 1 representation
        |
        +--> Layer 2 representation
        |
        v
Representation analysis
        |
        +--> PCA
        +--> Class mean activity
        +--> Cosine similarity
        +--> Nearest-centroid accuracy
        +--> Logistic regression accuracy
        +--> Per-class accuracy
"""

import os
import json

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

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
# Utility functions
# ============================================================

def nearest_centroid_predict(
    X_train,
    y_train,
    X_test,
):
    """
    Classify samples by nearest class centroid
    using Euclidean distance.
    """

    classes = np.unique(y_train)

    centroids = []

    for cls in classes:

        centroid = X_train[
            y_train == cls
        ].mean(axis=0)

        centroids.append(centroid)

    centroids = np.asarray(centroids)

    distances = np.linalg.norm(
        X_test[:, None, :] -
        centroids[None, :, :],
        axis=2,
    )

    predictions = classes[
        np.argmin(distances, axis=1)
    ]

    return predictions


def calculate_per_class_accuracy(
    y_true,
    y_pred,
    n_classes,
):
    """
    Calculate accuracy independently for every class.
    """

    result = {}

    for cls in range(n_classes):

        mask = y_true == cls

        if np.sum(mask) == 0:
            result[CLASS_NAMES[cls]] = None
            continue

        accuracy = np.mean(
            y_pred[mask] == y_true[mask]
        )

        result[CLASS_NAMES[cls]] = float(
            accuracy
        )

    return result


def plot_pca(
    X,
    y,
    layer_name,
    output_path,
):
    """
    PCA visualization of the representation.
    """

    pca = PCA(
        n_components=2,
    )

    X_pca = pca.fit_transform(X)

    explained = pca.explained_variance_ratio_

    plt.figure(
        figsize=(11, 8)
    )

    for cls in range(len(CLASS_NAMES)):

        mask = y == cls

        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=12,
            alpha=0.55,
            label=CLASS_NAMES[cls],
        )

    plt.xlabel(
        f"PC1 ({100 * explained[0]:.2f}%)"
    )

    plt.ylabel(
        f"PC2 ({100 * explained[1]:.2f}%)"
    )

    plt.title(
        f"STDP representation — {layer_name}"
    )

    plt.grid(
        alpha=0.25,
    )

    plt.legend(
        fontsize=8,
        ncol=2,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return explained.tolist()


def plot_class_mean_heatmap(
    X,
    y,
    layer_name,
    output_path,
):
    """
    Plot class x neuron mean firing-rate matrix.
    """

    n_classes = len(CLASS_NAMES)

    class_means = np.zeros(
        (
            n_classes,
            X.shape[1],
        ),
        dtype=np.float64,
    )

    for cls in range(n_classes):

        mask = y == cls

        if np.any(mask):

            class_means[cls] = X[
                mask
            ].mean(axis=0)

    plt.figure(
        figsize=(14, 8)
    )

    plt.imshow(
        class_means * 100.0,
        aspect="auto",
        interpolation="nearest",
    )

    plt.colorbar(
        label="Firing rate (%)"
    )

    plt.yticks(
        np.arange(n_classes),
        CLASS_NAMES,
    )

    plt.xlabel(
        "Neuron"
    )

    plt.ylabel(
        "SHD class"
    )

    plt.title(
        f"Class mean activity — {layer_name}"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return class_means


def plot_cosine_similarity(
    X,
    y,
    layer_name,
    output_path,
):
    """
    Cosine similarity between class-mean representations.
    """

    n_classes = len(CLASS_NAMES)

    class_means = np.zeros(
        (
            n_classes,
            X.shape[1],
        ),
        dtype=np.float64,
    )

    for cls in range(n_classes):

        mask = y == cls

        if np.any(mask):

            class_means[cls] = X[
                mask
            ].mean(axis=0)

    similarity = cosine_similarity(
        class_means
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(
        similarity,
        vmin=-1,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )

    plt.colorbar(
        label="Cosine similarity"
    )

    plt.xticks(
        np.arange(n_classes),
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        np.arange(n_classes),
        CLASS_NAMES,
    )

    plt.xlabel(
        "Class"
    )

    plt.ylabel(
        "Class"
    )

    plt.title(
        f"Class representation similarity — {layer_name}"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return similarity


def evaluate_representation(
    X,
    y,
    n_classes,
):
    """
    Evaluate classification quality of a representation.

    Two classifiers are used:

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

    centroid_per_class = (
        calculate_per_class_accuracy(
            y_test,
            centroid_pred,
            n_classes,
        )
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

    logistic_per_class = (
        calculate_per_class_accuracy(
            y_test,
            logistic_pred,
            n_classes,
        )
    )

    return {
        "nearest_centroid_accuracy":
            float(centroid_accuracy),

        "logistic_regression_accuracy":
            float(logistic_accuracy),

        "nearest_centroid_per_class":
            centroid_per_class,

        "logistic_regression_per_class":
            logistic_per_class,
    }


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
    print("STDP REPRESENTATION ANALYSIS")
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
    # Load STDP weights
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
    all_labels = []

    # --------------------------------------------------------
    # Run test set
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
            # Forward
            # ------------------------------------------------

            _, spikes1, spikes2 = model(
                spikes,
                apply_stdp=False,
                return_activity=True,
            )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Average only over time.
            #
            # [time, batch, neurons]
            #
            # -> [batch, neurons]
            #
            # Every neuron remains a separate feature.
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

            all_labels.append(
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
        all_labels,
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
    # Analyze layers
    # ========================================================

    analysis_results = {
        "checkpoint": stdp_checkpoint,
        "n_samples": int(len(y)),
        "n_classes": int(cfg.n_classes),
        "class_names": CLASS_NAMES,
    }

    for layer_name, X in [
        ("layer1", X1),
        ("layer2", X2),
    ]:

        pretty_name = (
            "Layer 1"
            if layer_name == "layer1"
            else "Layer 2"
        )

        print()
        print("=" * 60)
        print(
            f"{pretty_name} REPRESENTATION"
        )
        print("=" * 60)

        # ----------------------------------------------------
        # PCA
        # ----------------------------------------------------

        print(
            "\nRunning PCA..."
        )

        pca_path = os.path.join(
            checkpoint_dir,
            f"representation_pca_{layer_name}.png",
        )

        explained = plot_pca(
            X,
            y,
            pretty_name,
            pca_path,
        )

        print(
            f"PC1 explained variance : "
            f"{100 * explained[0]:.2f}%"
        )

        print(
            f"PC2 explained variance : "
            f"{100 * explained[1]:.2f}%"
        )

        # ----------------------------------------------------
        # Class mean activity
        # ----------------------------------------------------

        print(
            "\nCalculating class means..."
        )

        heatmap_path = os.path.join(
            checkpoint_dir,
            f"class_mean_activity_{layer_name}.png",
        )

        class_means = (
            plot_class_mean_heatmap(
                X,
                y,
                pretty_name,
                heatmap_path,
            )
        )

        # ----------------------------------------------------
        # Cosine similarity
        # ----------------------------------------------------

        print(
            "Calculating cosine similarity..."
        )

        similarity_path = os.path.join(
            checkpoint_dir,
            f"class_similarity_{layer_name}.png",
        )

        similarity = (
            plot_cosine_similarity(
                X,
                y,
                pretty_name,
                similarity_path,
            )
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        print(
            "\nEvaluating representation..."
        )

        classification = (
            evaluate_representation(
                X,
                y,
                cfg.n_classes,
            )
        )

        centroid_acc = (
            classification[
                "nearest_centroid_accuracy"
            ]
        )

        logistic_acc = (
            classification[
                "logistic_regression_accuracy"
            ]
        )

        print(
            f"\nNearest-centroid accuracy : "
            f"{100 * centroid_acc:.2f}%"
        )

        print(
            f"Logistic regression accuracy : "
            f"{100 * logistic_acc:.2f}%"
        )

        # ----------------------------------------------------
        # Per-class accuracy
        # ----------------------------------------------------

        print()
        print(
            "Logistic regression "
            "per-class accuracy:"
        )

        for cls, acc in (
            classification[
                "logistic_regression_per_class"
            ].items()
        ):

            if acc is None:

                print(
                    f"{cls:>5} : N/A"
                )

            else:

                print(
                    f"{cls:>5} : "
                    f"{100 * acc:6.2f}%"
                )

        # ----------------------------------------------------
        # Save layer results
        # ----------------------------------------------------

        analysis_results[layer_name] = {

            "n_features": int(
                X.shape[1]
            ),

            "pca_explained_variance": (
                explained
            ),

            "class_mean_activity": (
                class_means.tolist()
            ),

            "cosine_similarity": (
                similarity.tolist()
            ),

            "nearest_centroid_accuracy": (
                centroid_acc
            ),

            "logistic_regression_accuracy": (
                logistic_acc
            ),

            "nearest_centroid_per_class": (
                classification[
                    "nearest_centroid_per_class"
                ]
            ),

            "logistic_regression_per_class": (
                classification[
                    "logistic_regression_per_class"
                ]
            ),

            "plots": {
                "pca": pca_path,
                "class_mean_activity":
                    heatmap_path,
                "class_similarity":
                    similarity_path,
            },
        }

    # ========================================================
    # Final comparison
    # ========================================================

    print()
    print("=" * 60)
    print("LAYER COMPARISON")
    print("=" * 60)

    l1_centroid = (
        analysis_results[
            "layer1"
        ][
            "nearest_centroid_accuracy"
        ]
    )

    l2_centroid = (
        analysis_results[
            "layer2"
        ][
            "nearest_centroid_accuracy"
        ]
    )

    l1_logreg = (
        analysis_results[
            "layer1"
        ][
            "logistic_regression_accuracy"
        ]
    )

    l2_logreg = (
        analysis_results[
            "layer2"
        ][
            "logistic_regression_accuracy"
        ]
    )

    print(
        f"\n{'Metric':<30}"
        f"{'Layer 1':>12}"
        f"{'Layer 2':>12}"
    )

    print("-" * 54)

    print(
        f"{'Nearest centroid':<30}"
        f"{100 * l1_centroid:>11.2f}%"
        f"{100 * l2_centroid:>11.2f}%"
    )

    print(
        f"{'Logistic regression':<30}"
        f"{100 * l1_logreg:>11.2f}%"
        f"{100 * l2_logreg:>11.2f}%"
    )

    # ========================================================
    # Save JSON
    # ========================================================

    json_path = os.path.join(
        checkpoint_dir,
        "representation_analysis.json",
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            analysis_results,
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
        f"\nResults saved to:\n"
        f"{checkpoint_dir}"
    )

    print(
        f"\nJSON:\n"
        f"{json_path}"
    )


if __name__ == "__main__":
    main()