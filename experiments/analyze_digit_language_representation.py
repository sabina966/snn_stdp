"""
Analyze digit and language information in STDP representations.

SHD contains 20 classes:

    EN_0 ... EN_9
    DE_0 ... DE_9

This experiment evaluates whether the STDP representation contains:

1. LANGUAGE information:
       EN vs DE

2. DIGIT information:
       0 ... 9

Both are evaluated for:

    - Layer 1 full representation
    - Layer 2 full representation
    - Layer 1 population rate
    - Layer 2 population rate

The purpose is to determine whether the previously observed
20-class classification performance is driven mainly by:

    - language,
    - digit identity,
    - or both.
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
# Evaluation
# ============================================================

def evaluate_logistic(
    X,
    y,
):
    """
    Evaluate representation using logistic regression.

    A single fixed stratified split is used, matching the
    previous representation analysis.
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

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

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

    return float(accuracy)


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
        "STDP DIGIT / LANGUAGE REPRESENTATION ANALYSIS"
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

    layer1_features = []
    layer2_features = []

    layer1_population = []
    layer2_population = []

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
            # Forward
            # ------------------------------------------------

            _, spikes1, spikes2 = model(
                spikes,
                apply_stdp=False,
                return_activity=True,
            )

            # ------------------------------------------------
            # Mean firing rate per neuron
            #
            # [time, batch, neurons]
            #
            # ->
            #
            # [batch, neurons]
            # ------------------------------------------------

            rate1 = spikes1.float().mean(
                dim=0
            )

            rate2 = spikes2.float().mean(
                dim=0
            )

            # ------------------------------------------------
            # Population firing rate
            #
            # [batch]
            # ------------------------------------------------

            population1 = rate1.mean(
                dim=1
            )

            population2 = rate2.mean(
                dim=1
            )

            layer1_features.append(
                rate1.cpu().numpy()
            )

            layer2_features.append(
                rate2.cpu().numpy()
            )

            layer1_population.append(
                population1.cpu().numpy()
            )

            layer2_population.append(
                population2.cpu().numpy()
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

    P1 = np.concatenate(
        layer1_population,
        axis=0,
    ).reshape(
        -1,
        1,
    )

    P2 = np.concatenate(
        layer2_population,
        axis=0,
    ).reshape(
        -1,
        1,
    )

    y20 = np.concatenate(
        labels_all,
        axis=0,
    )

    print()
    print(
        f"Samples       : {len(y20)}"
    )

    print(
        f"Layer 1 shape : {X1.shape}"
    )

    print(
        f"Layer 2 shape : {X2.shape}"
    )

    # ========================================================
    # Construct DIGIT labels
    # ========================================================

    # EN_0 ... EN_9
    # DE_0 ... DE_9
    #
    # Original labels:
    #
    # 0 ... 9  -> English
    # 10 ... 19 -> German
    #
    # Digit is therefore:
    #
    # label % 10

    y_digit = (
        y20 % 10
    )

    # ========================================================
    # Construct LANGUAGE labels
    # ========================================================

    # 0 = English
    # 1 = German

    y_language = (
        y20 >= 10
    ).astype(
        np.int64
    )

    # ========================================================
    # Check distributions
    # ========================================================

    print()
    print("=" * 60)
    print(
        "CLASS DISTRIBUTION"
    )
    print("=" * 60)

    print()

    print(
        f"English samples : "
        f"{np.sum(y_language == 0)}"
    )

    print(
        f"German samples  : "
        f"{np.sum(y_language == 1)}"
    )

    print()

    for digit in range(10):

        count = np.sum(
            y_digit == digit
        )

        print(
            f"Digit {digit} : "
            f"{count}"
        )

    # ========================================================
    # Evaluation
    # ========================================================

    print()
    print("=" * 60)
    print(
        "LANGUAGE CLASSIFICATION"
    )
    print("=" * 60)

    print(
        "\nChance baseline : 50.00%"
    )

    # --------------------------------------------------------
    # Layer 1 full
    # --------------------------------------------------------

    l1_language = evaluate_logistic(
        X1,
        y_language,
    )

    print(
        f"\nLayer 1 full representation : "
        f"{100 * l1_language:.2f}%"
    )

    # --------------------------------------------------------
    # Layer 2 full
    # --------------------------------------------------------

    l2_language = evaluate_logistic(
        X2,
        y_language,
    )

    print(
        f"Layer 2 full representation : "
        f"{100 * l2_language:.2f}%"
    )

    # --------------------------------------------------------
    # Layer 1 population
    # --------------------------------------------------------

    p1_language = evaluate_logistic(
        P1,
        y_language,
    )

    print(
        f"Layer 1 population rate      : "
        f"{100 * p1_language:.2f}%"
    )

    # --------------------------------------------------------
    # Layer 2 population
    # --------------------------------------------------------

    p2_language = evaluate_logistic(
        P2,
        y_language,
    )

    print(
        f"Layer 2 population rate      : "
        f"{100 * p2_language:.2f}%"
    )

    # ========================================================
    # Digit classification
    # ========================================================

    print()
    print("=" * 60)
    print(
        "DIGIT CLASSIFICATION"
    )
    print("=" * 60)

    print(
        "\nChance baseline : 10.00%"
    )

    # --------------------------------------------------------
    # Layer 1 full
    # --------------------------------------------------------

    l1_digit = evaluate_logistic(
        X1,
        y_digit,
    )

    print(
        f"\nLayer 1 full representation : "
        f"{100 * l1_digit:.2f}%"
    )

    # --------------------------------------------------------
    # Layer 2 full
    # --------------------------------------------------------

    l2_digit = evaluate_logistic(
        X2,
        y_digit,
    )

    print(
        f"Layer 2 full representation : "
        f"{100 * l2_digit:.2f}%"
    )

    # --------------------------------------------------------
    # Layer 1 population
    # --------------------------------------------------------

    p1_digit = evaluate_logistic(
        P1,
        y_digit,
    )

    print(
        f"Layer 1 population rate      : "
        f"{100 * p1_digit:.2f}%"
    )

    # --------------------------------------------------------
    # Layer 2 population
    # --------------------------------------------------------

    p2_digit = evaluate_logistic(
        P2,
        y_digit,
    )

    print(
        f"Layer 2 population rate      : "
        f"{100 * p2_digit:.2f}%"
    )

    # ========================================================
    # Summary table
    # ========================================================

    print()
    print("=" * 60)
    print(
        "SUMMARY"
    )
    print("=" * 60)

    print()

    print(
        f"{'Representation':<32}"
        f"{'Language':>12}"
        f"{'Digit':>12}"
    )

    print(
        "-" * 56
    )

    print(
        f"{'Layer 1 full':<32}"
        f"{100 * l1_language:>11.2f}%"
        f"{100 * l1_digit:>11.2f}%"
    )

    print(
        f"{'Layer 1 population':<32}"
        f"{100 * p1_language:>11.2f}%"
        f"{100 * p1_digit:>11.2f}%"
    )

    print(
        f"{'Layer 2 full':<32}"
        f"{100 * l2_language:>11.2f}%"
        f"{100 * l2_digit:>11.2f}%"
    )

    print(
        f"{'Layer 2 population':<32}"
        f"{100 * p2_language:>11.2f}%"
        f"{100 * p2_digit:>11.2f}%"
    )

    # ========================================================
    # Per-language digit analysis
    # ========================================================

    print()
    print("=" * 60)
    print(
        "DIGIT CLASSIFICATION BY LANGUAGE"
    )
    print("=" * 60)

    print(
        "\nThis checks whether digit information"
    )

    print(
        "is equally accessible within EN and DE."
    )

    # --------------------------------------------------------
    # English only
    # --------------------------------------------------------

    english_mask = (
        y_language == 0
    )

    X1_en = X1[
        english_mask
    ]

    X2_en = X2[
        english_mask
    ]

    y_digit_en = y_digit[
        english_mask
    ]

    english_digit_l1 = evaluate_logistic(
        X1_en,
        y_digit_en,
    )

    english_digit_l2 = evaluate_logistic(
        X2_en,
        y_digit_en,
    )

    print()
    print(
        f"English — Layer 1 : "
        f"{100 * english_digit_l1:.2f}%"
    )

    print(
        f"English — Layer 2 : "
        f"{100 * english_digit_l2:.2f}%"
    )

    # --------------------------------------------------------
    # German only
    # --------------------------------------------------------

    german_mask = (
        y_language == 1
    )

    X1_de = X1[
        german_mask
    ]

    X2_de = X2[
        german_mask
    ]

    y_digit_de = y_digit[
        german_mask
    ]

    german_digit_l1 = evaluate_logistic(
        X1_de,
        y_digit_de,
    )

    german_digit_l2 = evaluate_logistic(
        X2_de,
        y_digit_de,
    )

    print(
        f"German  — Layer 1 : "
        f"{100 * german_digit_l1:.2f}%"
    )

    print(
        f"German  — Layer 2 : "
        f"{100 * german_digit_l2:.2f}%"
    )

    # ========================================================
    # Save
    # ========================================================

    results = {

        "checkpoint": stdp_checkpoint,

        "n_samples": int(
            len(y20)
        ),

        "language": {

            "chance_baseline": 0.50,

            "layer1_full":
                l1_language,

            "layer2_full":
                l2_language,

            "layer1_population":
                p1_language,

            "layer2_population":
                p2_language,
        },

        "digit": {

            "chance_baseline": 0.10,

            "layer1_full":
                l1_digit,

            "layer2_full":
                l2_digit,

            "layer1_population":
                p1_digit,

            "layer2_population":
                p2_digit,
        },

        "digit_by_language": {

            "english": {

                "layer1":
                    english_digit_l1,

                "layer2":
                    english_digit_l2,
            },

            "german": {

                "layer1":
                    german_digit_l1,

                "layer2":
                    german_digit_l2,
            },
        },
    }

    json_path = os.path.join(
        checkpoint_dir,
        "digit_language_analysis.json",
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
        "Digit/language analysis finished."
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