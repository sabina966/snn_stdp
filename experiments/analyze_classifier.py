"""
Analyze trained classifier on SHD.

This script does NOT train the model.

It evaluates:
    - overall accuracy
    - per-class accuracy
    - English accuracy
    - German accuracy
    - prediction distribution

The STDP layers remain frozen conceptually;
we only evaluate the already trained classifier.
"""

import os
import json

import torch

from configs import Config
from datasets.dataloader import get_dataloaders
from network.hierarchical_snn import HierarchicalSNN


# ============================================================
# Class names
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
    # Classifier checkpoint
    # --------------------------------------------------------

    classifier_checkpoint = input(
        "\nPath to classifier checkpoint:\n> "
    ).strip()

    if not os.path.exists(classifier_checkpoint):

        raise FileNotFoundError(
            f"\nClassifier checkpoint not found:\n"
            f"{classifier_checkpoint}"
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        cfg.device
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 60)
    print("Classifier analysis")
    print("=" * 60)

    print(f"Device     : {device}")
    print(f"Checkpoint : {classifier_checkpoint}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nLoading test dataset...")

    _, test_loader = get_dataloaders(
        root=cfg.dataset_root,
        batch_size=cfg.batch_size,
        time_steps=cfg.time_steps,
    )

    print("Dataset loaded.")

    # --------------------------------------------------------
    # Neuron parameters
    # --------------------------------------------------------

    neuron_params = dict(

        tau_m=cfg.tau_m,

        v_rest=cfg.v_rest,

        v_reset=cfg.v_reset,

        v_threshold=cfg.v_threshold,

        tau_adaptation=cfg.tau_adaptation,

        adaptation_strength=cfg.adaptation_strength,

    )

    # --------------------------------------------------------
    # STDP parameters
    # --------------------------------------------------------

    stdp_params = dict(

        a_plus=cfg.a_plus,

        a_minus=cfg.a_minus,

        tau_plus=cfg.tau_plus,

        tau_minus=cfg.tau_minus,

        w_min=cfg.w_min,

        w_max=cfg.w_max,

    )

    # --------------------------------------------------------
    # Homeostasis
    # --------------------------------------------------------

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

        use_classifier=True,

    ).to(device)

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    print("\nLoading classifier...")

    checkpoint = torch.load(
        classifier_checkpoint,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint,
        strict=True,
    )

    model.eval()

    print("Classifier loaded.")

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    correct = 0
    total = 0

    class_correct = [0] * cfg.n_classes
    class_total = [0] * cfg.n_classes

    prediction_counts = [0] * cfg.n_classes

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("\nEvaluating...\n")

    with torch.no_grad():

        for spikes, labels in test_loader:

            spikes = spikes.permute(
                1,
                0,
                2,
            ).to(device)

            labels = labels.to(device)

            logits = model(
                spikes,
                apply_stdp=False,
            )

            predictions = logits.argmax(
                dim=1
            )

            # Overall
            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            # Per class
            for label, prediction in zip(
                labels,
                predictions,
            ):

                label = label.item()
                prediction = prediction.item()

                class_total[label] += 1
                prediction_counts[prediction] += 1

                if label == prediction:

                    class_correct[label] += 1

    # --------------------------------------------------------
    # Overall accuracy
    # --------------------------------------------------------

    overall_accuracy = (
        correct / total
    )

    print("=" * 60)
    print("OVERALL")
    print("=" * 60)

    print(
        f"Accuracy: "
        f"{100 * overall_accuracy:.2f}%"
    )

    # --------------------------------------------------------
    # Per-class accuracy
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PER-CLASS ACCURACY")
    print("=" * 60)

    class_accuracy = {}

    for i in range(cfg.n_classes):

        if class_total[i] > 0:

            accuracy = (
                class_correct[i]
                / class_total[i]
            )

        else:

            accuracy = 0.0

        class_accuracy[CLASS_NAMES[i]] = accuracy

        print(
            f"{CLASS_NAMES[i]:>5} : "
            f"{100 * accuracy:6.2f}% "
            f"("
            f"{class_correct[i]}/"
            f"{class_total[i]}"
            f")"
        )

    # --------------------------------------------------------
    # English / German
    # --------------------------------------------------------

    english_correct = sum(
        class_correct[0:10]
    )

    english_total = sum(
        class_total[0:10]
    )

    german_correct = sum(
        class_correct[10:20]
    )

    german_total = sum(
        class_total[10:20]
    )

    english_accuracy = (
        english_correct / english_total
        if english_total > 0
        else 0.0
    )

    german_accuracy = (
        german_correct / german_total
        if german_total > 0
        else 0.0
    )

    print()
    print("=" * 60)
    print("LANGUAGE ACCURACY")
    print("=" * 60)

    print(
        f"English : "
        f"{100 * english_accuracy:.2f}%"
    )

    print(
        f"German  : "
        f"{100 * german_accuracy:.2f}%"
    )

    # --------------------------------------------------------
    # Prediction distribution
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PREDICTION DISTRIBUTION")
    print("=" * 60)

    for i in range(cfg.n_classes):

        percentage = (
            100
            * prediction_counts[i]
            / total
        )

        print(
            f"{CLASS_NAMES[i]:>5} : "
            f"{prediction_counts[i]:5d} "
            f"({percentage:6.2f}%)"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    analysis_dir = os.path.dirname(
        classifier_checkpoint
    )

    analysis_path = os.path.join(
        analysis_dir,
        "classifier_analysis.json",
    )

    results = {

        "overall_accuracy":
            overall_accuracy,

        "english_accuracy":
            english_accuracy,

        "german_accuracy":
            german_accuracy,

        "class_accuracy":
            class_accuracy,

        "class_correct":
            {
                CLASS_NAMES[i]:
                    class_correct[i]
                for i in range(cfg.n_classes)
            },

        "class_total":
            {
                CLASS_NAMES[i]:
                    class_total[i]
                for i in range(cfg.n_classes)
            },

        "prediction_distribution":
            {
                CLASS_NAMES[i]:
                    prediction_counts[i]
                for i in range(cfg.n_classes)
            },

    }

    with open(
        analysis_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print("=" * 60)
    print("Analysis finished.")
    print("=" * 60)

    print(
        f"\nSaved to:\n"
        f"{analysis_path}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()