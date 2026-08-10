"""
Train classifier on top of STDP-pretrained SNN.

Pipeline:

SHD
|
v
STDP-pretrained Layer 1
|
v
STDP-pretrained Layer 2
|
v
Spike decoder
|
v
Trainable classifier

STDP layers are frozen.
Only the classifier is trained.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from configs import Config
from datasets.dataloader import get_dataloaders
from network.hierarchical_snn import HierarchicalSNN

from utils.experiment import (
    create_run_directory,
    save_config,
    save_json,
)

from utils.plots import (
    plot_confusion_matrix,
    plot_digit_confusion_matrix,
    plot_loss,
    plot_accuracy,
)


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
            f"STDP checkpoint not found:\n"
            f"{stdp_checkpoint}"
        )

    # --------------------------------------------------------
    # Results directory
    # --------------------------------------------------------

    run_dir = create_run_directory(
        base="results",
        experiment="classifier",
    )

    save_config(
        cfg,
        os.path.join(
            run_dir,
            "config.json",
        ),
    )

    print()
    print("=" * 60)
    print("Classifier training on STDP weights")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        cfg.device
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device           : {device}")
    print(f"Epochs           : {cfg.epochs}")
    print(f"STDP checkpoint  : {stdp_checkpoint}")
    print(f"Results          : {run_dir}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_loader, test_loader = get_dataloaders(
        root=cfg.dataset_root,
        batch_size=cfg.batch_size,
        time_steps=cfg.time_steps,
    )

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

        use_classifier=True,

    ).to(device)

    # --------------------------------------------------------
    # Load STDP weights
    # --------------------------------------------------------

    print("\nLoading STDP weights...")

    checkpoint = torch.load(
        stdp_checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint,
        strict=True,
    )

    print("STDP weights loaded.")

    # --------------------------------------------------------
    # Freeze STDP layers
    # --------------------------------------------------------

    for param in model.layer1.parameters():

        param.requires_grad = False

    for param in model.layer2.parameters():

        param.requires_grad = False

    # --------------------------------------------------------
    # Make sure classifier is trainable
    # --------------------------------------------------------

    for param in model.classifier.parameters():

        param.requires_grad = True

    print("\nTrainable parameters:")

    for name, param in model.named_parameters():

        if param.requires_grad:

            print(
                f"  {name}: "
                f"{param.numel()} parameters"
            )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.Adam(

        model.classifier.parameters(),

        lr=cfg.learning_rate,

        weight_decay=cfg.weight_decay,

    )

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history = {

        "train_loss": [],

        "train_accuracy": [],

        "test_loss": [],

        "test_accuracy": [],

    }

    # --------------------------------------------------------
    # Best model tracking
    # --------------------------------------------------------

    best_test_accuracy = -1.0

    best_epoch = 0

    best_state_dict = None

    print("\nStarting classifier training...\n")

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(cfg.epochs):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        total_loss = 0.0

        correct = 0

        total = 0

        for spikes, labels in train_loader:

            spikes = spikes.permute(
                1,
                0,
                2,
            ).to(device)

            labels = labels.long().to(device)

            optimizer.zero_grad()

            logits = model(
                spikes,
                apply_stdp=False,
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = total_loss / total

        train_accuracy = correct / total

        # ====================================================
        # TEST
        # ====================================================

        model.eval()

        test_loss_total = 0.0

        test_correct = 0

        test_total = 0

        with torch.no_grad():

            for spikes, labels in test_loader:

                spikes = spikes.permute(
                    1,
                    0,
                    2,
                ).to(device)

                labels = labels.long().to(device)

                logits = model(
                    spikes,
                    apply_stdp=False,
                )

                loss = criterion(
                    logits,
                    labels,
                )

                test_loss_total += (
                    loss.item()
                    * labels.size(0)
                )

                predictions = logits.argmax(
                    dim=1
                )

                test_correct += (
                    predictions == labels
                ).sum().item()

                test_total += labels.size(0)

        test_loss = (
            test_loss_total
            / test_total
        )

        test_accuracy = (
            test_correct
            / test_total
        )

        # ====================================================
        # Save history
        # ====================================================

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["test_loss"].append(
            test_loss
        )

        history["test_accuracy"].append(
            test_accuracy
        )

        # ====================================================
        # Save best model
        # ====================================================

        if test_accuracy > best_test_accuracy:

            best_test_accuracy = test_accuracy

            best_epoch = epoch + 1

            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            print(
                f"\n*** New best model: "
                f"epoch {best_epoch}, "
                f"test accuracy = "
                f"{100 * best_test_accuracy:.2f}% ***"
            )

        # ====================================================
        # Print
        # ====================================================

        print()
        print("=" * 60)

        print(
            f"Epoch {epoch + 1}/{cfg.epochs}"
        )

        print(
            f"Train Loss : {train_loss:.4f}"
        )

        print(
            f"Train Acc  : "
            f"{100 * train_accuracy:.2f}%"
        )

        print(
            f"Test Loss  : {test_loss:.4f}"
        )

        print(
            f"Test Acc   : "
            f"{100 * test_accuracy:.2f}%"
        )

        print("=" * 60)

    # ========================================================
    # Save history
    # ========================================================

    save_json(
        history,
        os.path.join(
            run_dir,
            "history.json",
        ),
    )

    # ========================================================
    # Load best model
    # ========================================================

    model.load_state_dict(
        best_state_dict
    )

    model.eval()

    # ========================================================
    # Save best model
    # ========================================================

    best_model_path = os.path.join(
        run_dir,
        "best_model.pt",
    )

    torch.save(
        best_state_dict,
        best_model_path,
    )

    # ========================================================
    # Collect predictions from best model
    # ========================================================

    all_labels = []

    all_predictions = []

    with torch.no_grad():

        for spikes, labels in test_loader:

            spikes = spikes.permute(
                1,
                0,
                2,
            ).to(device)

            labels = labels.long().to(device)

            logits = model(
                spikes,
                apply_stdp=False,
            )

            predictions = logits.argmax(
                dim=1
            )

            all_labels.append(
                labels.cpu()
            )

            all_predictions.append(
                predictions.cpu()
            )

    all_labels = torch.cat(
        all_labels
    )

    all_predictions = torch.cat(
        all_predictions
    )

    # ========================================================
    # Save best predictions
    # ========================================================

    predictions_path = os.path.join(
        run_dir,
        "best_predictions.pt",
    )

    torch.save(
        {
            "labels": all_labels,
            "predictions": all_predictions,
            "epoch": best_epoch,
            "accuracy": best_test_accuracy,
        },
        predictions_path,
    )

    # ========================================================
    # Confusion matrices
    # ========================================================

    plot_confusion_matrix(

        labels=all_labels,

        predictions=all_predictions,

        save_path=os.path.join(
            run_dir,
            "confusion_matrix.png",
        ),

        n_classes=cfg.n_classes,

    )

    plot_digit_confusion_matrix(

        labels=all_labels,

        predictions=all_predictions,

        save_path=os.path.join(
            run_dir,
            "confusion_matrix_digits.png",
        ),

    )

    # ========================================================
    # Loss and accuracy plots
    # ========================================================

    plot_loss(
        history,
        os.path.join(
            run_dir,
            "loss.png",
        ),
    )

    plot_accuracy(
        history,
        os.path.join(
            run_dir,
            "accuracy.png",
        ),
    )

    # ========================================================
    # Metrics
    # ========================================================

    metrics = {

        "final_train_loss":
            history["train_loss"][-1],

        "final_train_accuracy":
            history["train_accuracy"][-1],

        "final_test_loss":
            history["test_loss"][-1],

        "final_test_accuracy":
            history["test_accuracy"][-1],

        "best_test_accuracy":
            best_test_accuracy,

        "best_epoch":
            best_epoch,

        "best_model":
            "best_model.pt",

        "best_predictions":
            "best_predictions.pt",

    }

    save_json(
        metrics,
        os.path.join(
            run_dir,
            "metrics.json",
        ),
    )

    # ========================================================
    # Finished
    # ========================================================

    print()
    print("=" * 60)
    print("Finished.")
    print("=" * 60)

    print(
        f"Best test accuracy: "
        f"{100 * best_test_accuracy:.2f}%"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Results saved to: "
        f"{run_dir}"
    )

    print(
        f"Best model saved to: "
        f"{best_model_path}"
    )

    print(
        f"Predictions saved to: "
        f"{predictions_path}"
    )


if __name__ == "__main__":

    main()

