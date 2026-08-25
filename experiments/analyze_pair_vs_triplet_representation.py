"""
Compare representations produced by controlled Pair-STDP
and Triplet-STDP networks.

IMPORTANT:
- Does NOT perform STDP.
- Does NOT modify saved weights.
- Uses exactly the same SHD samples for Pair and Triplet.
- Loads weights from a controlled experiment.
"""

import os
import sys
import json
import random
from datetime import datetime

import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from datasets.shd import SHDDataset
from network.hierarchical_snn import HierarchicalSNN
from network.triplet.triplet_snn import TripletHierarchicalSNN


# ============================================================
# CONFIG
# ============================================================

TIME_STEPS = 200
N_INPUT = 700
N_HIDDEN1 = 200
N_HIDDEN2 = 100
N_CLASSES = 20

BATCH_SIZE = 32

# Number of SHD test samples used for representation analysis.
# Increase later if the run is fast enough.
MAX_SAMPLES = 1000

SEED = 42

DEVICE = torch.device("cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# UTILITIES
# ============================================================

def load_tensor(path):
    return torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )


def mean_activity(spikes):
    """
    Convert spike tensor:

        [time, batch, neurons]

    into:

        [batch, neurons]

    using mean firing activity over time.
    """

    return spikes.float().mean(dim=0)


def language_labels(labels):
    """
    SHD labels:

        0-9   -> English
        10-19 -> German

    Returns:
        0 -> English
        1 -> German
    """

    return (labels >= 10).long()


def digit_labels(labels):
    """
    Convert SHD labels 0-19 into digit labels 0-9.

    English:
        0-9

    German:
        10-19

    Therefore:

        digit = label % 10
    """

    return labels % 10


# ============================================================
# MODEL CREATION
# ============================================================

def create_pair_model():
    """
    Create original Pair-STDP network.

    STDP will NOT be applied during representation analysis.
    """

    model = HierarchicalSNN(
        n_input=N_INPUT,
        n_hidden1=N_HIDDEN1,
        n_hidden2=N_HIDDEN2,
        n_classes=N_CLASSES,

        neuron_params={
            "tau_m": 20.0,
            "v_rest": -65.0,
            "v_reset": -65.0,
            "v_threshold": -50.0,
        },

        stdp_params1={
            "a_plus": 0.001,
            "a_minus": 0.0012,
            "tau_plus": 20.0,
            "tau_minus": 20.0,
        },

        stdp_params2={
            "a_plus": 0.001,
            "a_minus": 0.0012,
            "tau_plus": 20.0,
            "tau_minus": 20.0,
        },

        homeostasis_params={
            "target_rate": 0.05,
            "tau_homeostasis": 5000.0,
            "strength": 0.02,
        },

        input_gain=15.0,
        use_classifier=False,
    )

    return model


def create_triplet_model():
    """
    Create Triplet-STDP network.

    STDP will NOT be applied during representation analysis.
    """

    model = TripletHierarchicalSNN(
        n_input=N_INPUT,
        n_hidden1=N_HIDDEN1,
        n_hidden2=N_HIDDEN2,
        n_classes=N_CLASSES,

        neuron_params={
            "tau_m": 20.0,
            "v_rest": -65.0,
            "v_reset": -65.0,
            "v_threshold": -50.0,
        },

        triplet_params1={
            "a2_plus": 0.001,
            "a2_minus": 0.0012,
            "a3_plus": 0.001,
            "a3_minus": 0.001,
            "tau_plus": 20.0,
            "tau_minus": 20.0,
            "tau_x": 100.0,
            "tau_y": 100.0,
        },

        triplet_params2={
            "a2_plus": 0.001,
            "a2_minus": 0.0012,
            "a3_plus": 0.001,
            "a3_minus": 0.001,
            "tau_plus": 20.0,
            "tau_minus": 20.0,
            "tau_x": 100.0,
            "tau_y": 100.0,
        },

        homeostasis_params={
            "target_rate": 0.05,
            "tau_homeostasis": 5000.0,
            "strength": 0.02,
        },

        input_gain=15.0,
        use_classifier=False,
    )

    return model


# ============================================================
# LOAD WEIGHTS
# ============================================================

def load_pair_weights(model, results_dir):

    model.layer1.weights.data.copy_(
        load_tensor(
            os.path.join(
                results_dir,
                "pair_final_weights_layer1.pt",
            )
        )
    )

    model.layer2.weights.data.copy_(
        load_tensor(
            os.path.join(
                results_dir,
                "pair_final_weights_layer2.pt",
            )
        )
    )


def load_triplet_weights(model, results_dir):

    model.layer1.weights.data.copy_(
        load_tensor(
            os.path.join(
                results_dir,
                "triplet_final_weights_layer1.pt",
            )
        )
    )

    model.layer2.weights.data.copy_(
        load_tensor(
            os.path.join(
                results_dir,
                "triplet_final_weights_layer2.pt",
            )
        )
    )


# ============================================================
# DATA
# ============================================================

def load_test_data():

    print()
    print("=" * 60)
    print("LOADING SHD TEST DATA")
    print("=" * 60)

    dataset = SHDDataset(
        root="./data",
        train=False,
        time_steps=TIME_STEPS,
        n_input=N_INPUT,
    )

    n_samples = min(
        MAX_SAMPLES,
        len(dataset),
    )

    print("Available test samples :", len(dataset))
    print("Samples used           :", n_samples)

    spikes_list = []
    labels_list = []

    for i in range(n_samples):

        spikes, label = dataset[i]

        spikes_list.append(spikes)
        labels_list.append(label)

    spikes = torch.stack(
        spikes_list
    )

    labels = torch.stack(
        labels_list
    )

    print("Dataset tensor :", tuple(spikes.shape))
    print("Labels tensor  :", tuple(labels.shape))

    return spikes, labels


# ============================================================
# REPRESENTATION EXTRACTION
# ============================================================

@torch.no_grad()
def extract_representation(
    model,
    spikes,
):
    """
    Extract Layer 1 and Layer 2 activity.

    No STDP is applied.
    """

    model.eval()

    layer1_activity = []
    layer2_activity = []

    n_samples = spikes.shape[0]

    for start in range(
        0,
        n_samples,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            n_samples,
        )

        batch = spikes[start:end]

        # Dataset:
        # [batch, time, input]
        #
        # Network:
        # [time, batch, input]

        batch = batch.permute(
            1,
            0,
            2,
        ).to(DEVICE)

        spikes1, spikes2 = model(
            batch,
            apply_stdp=False,
            return_activity=True,
        )

        activity1 = mean_activity(
            spikes1
        )

        activity2 = mean_activity(
            spikes2
        )

        layer1_activity.append(
            activity1.cpu()
        )

        layer2_activity.append(
            activity2.cpu()
        )

        print(
            f"\rProcessed {end}/{n_samples}",
            end="",
        )

    print()

    layer1 = torch.cat(
        layer1_activity,
        dim=0,
    ).numpy()

    layer2 = torch.cat(
        layer2_activity,
        dim=0,
    ).numpy()

    return layer1, layer2


# ============================================================
# PCA
# ============================================================

def run_pca(X):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    pca = PCA(
        n_components=2,
        random_state=SEED,
    )

    X_pca = pca.fit_transform(
        X_scaled
    )

    explained = (
        pca.explained_variance_ratio_
    )

    return X_pca, explained


# ============================================================
# CLASSIFICATION
# ============================================================

def logistic_accuracy(
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Logistic regression accuracy.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    clf = LogisticRegression(
        max_iter=1000,
        random_state=SEED,
    )

    clf.fit(
        X_train_scaled,
        y_train,
    )

    prediction = clf.predict(
        X_test_scaled
    )

    return accuracy_score(
        y_test,
        prediction,
    )


def nearest_centroid_accuracy(
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Nearest centroid classifier.
    """

    classes = np.unique(
        y_train
    )

    centroids = {}

    for c in classes:

        centroids[c] = X_train[
            y_train == c
        ].mean(axis=0)

    predictions = []

    for sample in X_test:

        distances = {}

        for c in classes:

            distances[c] = np.linalg.norm(
                sample - centroids[c]
            )

        prediction = min(
            distances,
            key=distances.get,
        )

        predictions.append(
            prediction
        )

    return accuracy_score(
        y_test,
        predictions,
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(
    X,
    labels,
):
    """
    Fixed deterministic 70/30 split.
    """

    n = len(X)

    rng = np.random.RandomState(
        SEED
    )

    indices = np.arange(n)

    rng.shuffle(
        indices
    )

    split = int(
        0.7 * n
    )

    train_idx = indices[
        :split
    ]

    test_idx = indices[
        split:
    ]

    return (
        X[train_idx],
        X[test_idx],
        labels[train_idx],
        labels[test_idx],
    )


# ============================================================
# PER-DIGIT ACCURACY
# ============================================================

def per_class_accuracy(
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Logistic regression per-class accuracy.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    clf = LogisticRegression(
        max_iter=1000,
        random_state=SEED,
    )

    clf.fit(
        X_train_scaled,
        y_train,
    )

    prediction = clf.predict(
        X_test_scaled
    )

    result = {}

    for c in sorted(
        np.unique(y_test)
    ):

        mask = (
            y_test == c
        )

        result[str(int(c))] = float(
            accuracy_score(
                y_test[mask],
                prediction[mask],
            )
        )

    return result


# ============================================================
# ANALYZE ONE REPRESENTATION
# ============================================================

def analyze_representation(
    X,
    labels,
    name,
):
    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(
        "Samples   :",
        X.shape[0],
    )

    print(
        "Neurons   :",
        X.shape[1],
    )

    X_train, X_test, y_train, y_test = (
        split_data(
            X,
            labels,
        )
    )

    # --------------------------------------------------------
    # Digit labels
    # --------------------------------------------------------

    digit_train = digit_labels(
        torch.tensor(y_train)
    ).numpy()

    digit_test = digit_labels(
        torch.tensor(y_test)
    ).numpy()

    # --------------------------------------------------------
    # Language labels
    # --------------------------------------------------------

    language_train = language_labels(
        torch.tensor(y_train)
    ).numpy()

    language_test = language_labels(
        torch.tensor(y_test)
    ).numpy()

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    X_pca, explained = run_pca(
        X
    )

    print()
    print("PCA")
    print(
        f"PC1 explained variance : "
        f"{explained[0] * 100:.2f}%"
    )

    print(
        f"PC2 explained variance : "
        f"{explained[1] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Digit classification
    # --------------------------------------------------------

    digit_logistic = logistic_accuracy(
        X_train,
        digit_train,
        X_test,
        digit_test,
    )

    digit_centroid = nearest_centroid_accuracy(
        X_train,
        digit_train,
        X_test,
        digit_test,
    )

    # --------------------------------------------------------
    # Language classification
    # --------------------------------------------------------

    language_logistic = logistic_accuracy(
        X_train,
        language_train,
        X_test,
        language_test,
    )

    language_centroid = nearest_centroid_accuracy(
        X_train,
        language_train,
        X_test,
        language_test,
    )

    print()
    print("DIGIT")
    print(
        f"Nearest centroid : "
        f"{digit_centroid * 100:.2f}%"
    )

    print(
        f"Logistic          : "
        f"{digit_logistic * 100:.2f}%"
    )

    print()
    print("LANGUAGE")
    print(
        f"Nearest centroid : "
        f"{language_centroid * 100:.2f}%"
    )

    print(
        f"Logistic          : "
        f"{language_logistic * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Per digit
    # --------------------------------------------------------

    per_digit = per_class_accuracy(
        X_train,
        digit_train,
        X_test,
        digit_test,
    )

    print()
    print("PER-DIGIT LOGISTIC ACCURACY")

    for digit, acc in per_digit.items():

        print(
            f"Digit {digit}: "
            f"{acc * 100:.2f}%"
        )

    return {
        "samples": int(X.shape[0]),
        "neurons": int(X.shape[1]),

        "pca_pc1": float(
            explained[0]
        ),

        "pca_pc2": float(
            explained[1]
        ),

        "digit": {
            "nearest_centroid": float(
                digit_centroid
            ),
            "logistic": float(
                digit_logistic
            ),
        },

        "language": {
            "nearest_centroid": float(
                language_centroid
            ),
            "logistic": float(
                language_logistic
            ),
        },

        "per_digit": per_digit,
    }, X_pca


# ============================================================
# PLOT PCA
# ============================================================

def plot_pca(
    X_pca,
    labels,
    title,
    path,
):
    plt.figure(
        figsize=(8, 6)
    )

    digits = (
        labels % 10
    )

    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=digits,
        s=12,
        alpha=0.65,
    )

    plt.xlabel(
        "PC1"
    )

    plt.ylabel(
        "PC2"
    )

    plt.title(
        title
    )

    plt.colorbar(
        scatter,
        label="Digit",
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()


# ============================================================
# PLOT COMPARISON
# ============================================================

def plot_accuracy_comparison(
    results,
    path,
):
    names = [
        "Pair L1",
        "Triplet L1",
        "Pair L2",
        "Triplet L2",
    ]

    values = [
        results["pair_layer1"]["digit"]["logistic"],
        results["triplet_layer1"]["digit"]["logistic"],
        results["pair_layer2"]["digit"]["logistic"],
        results["triplet_layer2"]["digit"]["logistic"],
    ]

    values = [
        v * 100
        for v in values
    ]

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        names,
        values,
    )

    plt.ylabel(
        "Digit accuracy (%)"
    )

    plt.title(
        "Pair vs Triplet STDP representation"
    )

    plt.ylim(
        0,
        max(
            max(values) * 1.2,
            15,
        ),
    )

    for i, value in enumerate(
        values
    ):

        plt.text(
            i,
            value,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()


def plot_language_comparison(
    results,
    path,
):
    names = [
        "Pair L1",
        "Triplet L1",
        "Pair L2",
        "Triplet L2",
    ]

    values = [
        results["pair_layer1"]["language"]["logistic"],
        results["triplet_layer1"]["language"]["logistic"],
        results["pair_layer2"]["language"]["logistic"],
        results["triplet_layer2"]["language"]["logistic"],
    ]

    values = [
        v * 100
        for v in values
    ]

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        names,
        values,
    )

    plt.axhline(
        50,
        linestyle="--",
        label="Chance = 50%",
    )

    plt.ylabel(
        "Language accuracy (%)"
    )

    plt.title(
        "Pair vs Triplet: language information"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "poetry run python "
            "experiments/analyze_pair_vs_triplet_representation.py "
            "results/controlled_pair_triplet/YYYYMMDD_HHMMSS"
        )

        sys.exit(1)

    results_dir = sys.argv[1]

    if not os.path.isdir(
        results_dir
    ):

        raise FileNotFoundError(
            f"Results directory not found:\n"
            f"{results_dir}"
        )

    set_seed(
        SEED
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_dir = os.path.join(
        "results",
        "pair_vs_triplet_representation",
        timestamp,
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    print("=" * 60)
    print(
        "PAIR VS TRIPLET REPRESENTATION ANALYSIS"
    )
    print("=" * 60)

    print()
    print(
        "Controlled experiment:"
    )

    print(
        results_dir
    )

    print()
    print(
        "Output:"
    )

    print(
        output_dir
    )

    print()
    print(
        "Device:",
        DEVICE,
    )

    # ========================================================
    # DATA
    # ========================================================

    spikes, labels = load_test_data()

    # ========================================================
    # MODELS
    # ========================================================

    print()
    print(
        "=" * 60
    )
    print(
        "CREATING MODELS"
    )
    print(
        "=" * 60
    )

    pair_model = create_pair_model()

    triplet_model = create_triplet_model()

    pair_model.to(
        DEVICE
    )

    triplet_model.to(
        DEVICE
    )

    # ========================================================
    # LOAD CONTROLLED WEIGHTS
    # ========================================================

    print()
    print(
        "Loading Pair weights..."
    )

    load_pair_weights(
        pair_model,
        results_dir,
    )

    print(
        "Loading Triplet weights..."
    )

    load_triplet_weights(
        triplet_model,
        results_dir,
    )

    print(
        "Weights loaded successfully."
    )

    # ========================================================
    # EXTRACT REPRESENTATIONS
    # ========================================================

    print()
    print(
        "=" * 60
    )
    print(
        "PAIR REPRESENTATION"
    )
    print(
        "=" * 60
    )

    pair_l1, pair_l2 = (
        extract_representation(
            pair_model,
            spikes,
        )
    )

    print(
        "Pair Layer 1:",
        pair_l1.shape,
    )

    print(
        "Pair Layer 2:",
        pair_l2.shape,
    )

    print()
    print(
        "=" * 60
    )
    print(
        "TRIPLET REPRESENTATION"
    )
    print(
        "=" * 60
    )

    triplet_l1, triplet_l2 = (
        extract_representation(
            triplet_model,
            spikes,
        )
    )

    print(
        "Triplet Layer 1:",
        triplet_l1.shape,
    )

    print(
        "Triplet Layer 2:",
        triplet_l2.shape,
    )

    # ========================================================
    # ANALYSIS
    # ========================================================

    results = {}

    pair_l1_result, pair_l1_pca = (
        analyze_representation(
            pair_l1,
            labels.numpy(),
            "PAIR — LAYER 1",
        )
    )

    results[
        "pair_layer1"
    ] = pair_l1_result

    pair_l2_result, pair_l2_pca = (
        analyze_representation(
            pair_l2,
            labels.numpy(),
            "PAIR — LAYER 2",
        )
    )

    results[
        "pair_layer2"
    ] = pair_l2_result

    triplet_l1_result, triplet_l1_pca = (
        analyze_representation(
            triplet_l1,
            labels.numpy(),
            "TRIPLET — LAYER 1",
        )
    )

    results[
        "triplet_layer1"
    ] = triplet_l1_result

    triplet_l2_result, triplet_l2_pca = (
        analyze_representation(
            triplet_l2,
            labels.numpy(),
            "TRIPLET — LAYER 2",
        )
    )

    results[
        "triplet_layer2"
    ] = triplet_l2_result

    # ========================================================
    # PCA PLOTS
    # ========================================================

    plot_pca(
        pair_l1_pca,
        labels.numpy(),
        "Pair STDP — Layer 1",
        os.path.join(
            output_dir,
            "pca_pair_layer1.png",
        ),
    )

    plot_pca(
        triplet_l1_pca,
        labels.numpy(),
        "Triplet STDP — Layer 1",
        os.path.join(
            output_dir,
            "pca_triplet_layer1.png",
        ),
    )

    plot_pca(
        pair_l2_pca,
        labels.numpy(),
        "Pair STDP — Layer 2",
        os.path.join(
            output_dir,
            "pca_pair_layer2.png",
        ),
    )

    plot_pca(
        triplet_l2_pca,
        labels.numpy(),
        "Triplet STDP — Layer 2",
        os.path.join(
            output_dir,
            "pca_triplet_layer2.png",
        ),
    )

    # ========================================================
    # COMPARISON PLOTS
    # ========================================================

    plot_accuracy_comparison(
        results,
        os.path.join(
            output_dir,
            "digit_accuracy.png",
        ),
    )

    plot_language_comparison(
        results,
        os.path.join(
            output_dir,
            "language_accuracy.png",
        ),
    )

    # ========================================================
    # SAVE JSON
    # ========================================================

    summary = {
        "config": {
            "time_steps": TIME_STEPS,
            "n_input": N_INPUT,
            "hidden1": N_HIDDEN1,
            "hidden2": N_HIDDEN2,
            "batch_size": BATCH_SIZE,
            "max_samples": MAX_SAMPLES,
            "seed": SEED,
            "controlled_experiment": results_dir,
        },

        "results": results,
    }

    with open(
        os.path.join(
            output_dir,
            "summary.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print(
        "FINAL COMPARISON"
    )
    print("=" * 60)

    print()

    print(
        "DIGIT — LOGISTIC REGRESSION"
    )

    print(
        f"Pair Layer 1    : "
        f"{results['pair_layer1']['digit']['logistic'] * 100:.2f}%"
    )

    print(
        f"Triplet Layer 1 : "
        f"{results['triplet_layer1']['digit']['logistic'] * 100:.2f}%"
    )

    print(
        f"Pair Layer 2    : "
        f"{results['pair_layer2']['digit']['logistic'] * 100:.2f}%"
    )

    print(
        f"Triplet Layer 2 : "
        f"{results['triplet_layer2']['digit']['logistic'] * 100:.2f}%"
    )

    print()

    print(
        "LANGUAGE — LOGISTIC REGRESSION"
    )

    print(
        f"Pair Layer 1    : "
        f"{results['pair_layer1']['language']['logistic'] * 100:.2f}%"
    )

    print(
        f"Triplet Layer 1 : "
        f"{results['triplet_layer1']['language']['logistic'] * 100:.2f}%"
    )

    print(
        f"Pair Layer 2    : "
        f"{results['pair_layer2']['language']['logistic'] * 100:.2f}%"
    )

    print(
        f"Triplet Layer 2 : "
        f"{results['triplet_layer2']['language']['logistic'] * 100:.2f}%"
    )

    print()
    print("=" * 60)
    print(
        "REPRESENTATION ANALYSIS FINISHED"
    )
    print("=" * 60)

    print()
    print(
        "Results saved to:"
    )

    print(
        output_dir
    )


if __name__ == "__main__":
    main()