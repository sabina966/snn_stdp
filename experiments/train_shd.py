"""
Train Hierarchical SNN on SHD.
"""

import os
import json

from utils.experiment import (
    create_run_directory,
    save_json,
)
from utils.plots import (
    plot_raster,
    plot_confusion_matrix,
    plot_digit_confusion_matrix,
    plot_weights,
    plot_loss,
    plot_accuracy,
    plot_firing_rate,
    plot_weight_histogram,
)

from utils.analysis import (
    weight_statistics,
    summarize_history,
)

import torch
import torch.optim as optim

from configs import Config
from datasets import get_dataloaders
from network.hierarchical_snn import HierarchicalSNN
from training import Trainer


def main():

    cfg = Config()
    run_dir = create_run_directory()

    print(
        f"\nExperiment directory:"
    )
    print(run_dir)

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available() and cfg.device == "cuda"
        else "cpu"
    )

    print("=" * 60)
    print("Hierarchical SNN")
    print("=" * 60)
    print(f"Device      : {device}")
    print(f"Batch size  : {cfg.batch_size}")
    print(f"Epochs      : {cfg.epochs}")
    print(f"Time steps  : {cfg.time_steps}")
    print("=" * 60)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    print("\nLoading dataset...")

    train_loader, test_loader = get_dataloaders(
        root=cfg.dataset_root,
        batch_size=cfg.batch_size,
        time_steps=cfg.time_steps,
    )

    print("Done.")

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    print("\nBuilding network...")

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
    )

    print(model)

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=device,
    )

    # --------------------------------------------------
    # STDP pretraining
    # --------------------------------------------------

    if cfg.stdp_pretrain_epochs > 0:

        print("\nStarting STDP pretraining...")

        for epoch in range(cfg.stdp_pretrain_epochs):

            trainer.stdp_pretrain_epoch()

            print(f"STDP epoch {epoch + 1}/{cfg.stdp_pretrain_epochs}")

        print("STDP pretraining finished.\n")
        
        print("Layer 1 mean weight:",
            model.layer1.weights.mean().item())

        print("Layer 2 mean weight:",
            model.layer2.weights.mean().item())
        
    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    fit_result = trainer.fit(
        cfg.epochs
    )

    history = fit_result["history"]

    best_epoch = fit_result["best_epoch"]
    best_test_accuracy = fit_result["best_test_accuracy"]

    best_labels = fit_result["best_labels"]
    best_predictions = fit_result["best_predictions"]

    best_state_dict = fit_result["best_state_dict"]

    history = {
        k: [float(x) for x in v]
        for k, v in history.items()
    }

    # --------------------------------------------------
    # Save best model
    # --------------------------------------------------

    torch.save(
        best_state_dict,
        f"{run_dir}/best_model.pt"
    )

    # --------------------------------------------------
    # Save best predictions
    # --------------------------------------------------

    torch.save(
        {
            "labels": best_labels,
            "predictions": best_predictions,
            "epoch": best_epoch,
            "accuracy": best_test_accuracy,
        },
        f"{run_dir}/best_predictions.pt",
    )

    print("\nBest model:")
    print(f"Epoch    : {best_epoch}")
    print(f"Accuracy : {100 * best_test_accuracy:.2f}%")

    # --------------------------------------------------
    # Load best model for visualization
    # --------------------------------------------------

    model.load_state_dict(best_state_dict)
    model.eval()

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------

    spikes, labels = next(iter(test_loader))

    spikes = spikes.permute(1, 0, 2).to(device)

    with torch.no_grad():

        _, layer1, layer2 = model(
            spikes,
            return_activity=True,
        )

    plot_raster(
        layer1,
        f"{run_dir}/raster_layer1.png",
        title="Спайкавая актыўнасць першага слоя",
    )

    plot_raster(
        layer2,
        f"{run_dir}/raster_layer2.png",
        title="Спайкавая актыўнасць другога слоя",
    )

    plot_weights(
        model.layer1.weights,
        f"{run_dir}/weights_layer1.png",
        title="Вагі сінапсаў першага слоя",
    )

    plot_weight_histogram(
        model.layer1.weights,
        f"{run_dir}/weight_hist_layer1.png",
        title="Размеркаванне ваг першага слоя",
    )

    plot_weights(
        model.layer2.weights,
        f"{run_dir}/weights_layer2.png",
        title="Вагі сінапсаў другога слоя",
    )

    plot_weight_histogram(
        model.layer2.weights,
        f"{run_dir}/weight_hist_layer2.png",
        title="Размеркаванне ваг другога слоя",
    )


    # --------------------------------------------------
    # Confusion matrix for best model
    # --------------------------------------------------

    plot_confusion_matrix(
        labels=best_labels,
        predictions=best_predictions,
        save_path=f"{run_dir}/confusion_matrix_best.png",
        n_classes=cfg.n_classes,
    )

    plot_digit_confusion_matrix(
        labels=best_labels,
        predictions=best_predictions,
        save_path=f"{run_dir}/confusion_matrix_digits_best.png",
    )  


    # --------------------------------------------------
    # Weight statistics
    # --------------------------------------------------

    layer1_stats = weight_statistics(
        model.layer1.weights
    )

    layer2_stats = weight_statistics(
        model.layer2.weights
    )

    weight_stats = {
        "layer1": layer1_stats,
        "layer2": layer2_stats,
    }

    save_json(
        weight_stats,
        f"{run_dir}/weight_statistics.json",
    )

    summary = summarize_history(
        history
    )

    summary["best_epoch"] = best_epoch
    summary["best_test_accuracy"] = float(best_test_accuracy)
    summary["best_model"] = "best_model.pt"
    summary["best_predictions"] = "best_predictions.pt"

    save_json(
        summary,
        f"{run_dir}/summary.json",
    )

    print("\nLayer 1 weight statistics")
    print(layer1_stats)

    print("\nLayer 2 weight statistics")
    print(layer2_stats)

    save_json(
        history,
        f"{run_dir}/history.json"
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    trainer.save(
        f"{run_dir}/model.pt"
    )


    config_dict = vars(cfg)

    save_json(
        config_dict,
        f"{run_dir}/config.json"
    )

    plot_loss(
        history,
        f"{run_dir}/loss.png",
    )

    plot_accuracy(
        history,
        f"{run_dir}/accuracy.png",
    )

    plot_firing_rate(
        history,
        f"{run_dir}/firing_rate.png",
    )
    print("\nModel saved to:")
    print(f"{run_dir}/model.pt")

    print("\nConfig saved to:")
    print(f"{run_dir}/config.json")


    print("\nHistory saved to:")
    print(f"{run_dir}/history.json")

    print("\nWeight statistics saved:")
    print(f"{run_dir}/weight_statistics.json")

    print("\nSummary saved:")
    print(f"{run_dir}/summary.json")

    print("\nBest test accuracy:")
    print(f"{100 * summary['best_test_accuracy']:.2f}%")

    print("Best epoch:")
    print(summary["best_epoch"])


    return history

if __name__ == "__main__":
    main()