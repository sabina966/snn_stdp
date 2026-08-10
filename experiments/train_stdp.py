"""
Pure STDP training.

Pipeline

SHD
|
Input spikes
|
HierarchicalSNN
|
Pair / Triplet STDP
|
Analysis
|
Save learned weights
"""

import os
import torch

from configs import Config
from datasets.dataloader import get_dataloaders
from network.hierarchical_snn import HierarchicalSNN

from utils.stdp_analysis import (
    analyze_stdp,
    get_weight_statistics,
)

from utils.experiment import (
    create_run_directory,
    save_config,
)


def main():

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    cfg = Config()

    # Create unique directory for this experiment
    run_dir = create_run_directory(
        base="results",
        experiment="stdp",
    )

    # Save configuration
    save_config(
        cfg,
        os.path.join(
            run_dir,
            "config.json",
        ),
    )

    print(f"Results directory: {run_dir}")

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        cfg.device
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("Pure STDP training")
    print("=" * 60)
    print(f"Device     : {device}")
    print(f"Epochs     : {cfg.stdp_pretrain_epochs}")
    print(f"Time steps : {cfg.time_steps}")
    print("=" * 60)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    train_loader, _ = get_dataloaders(
        root=cfg.dataset_root,
        batch_size=cfg.batch_size,
        time_steps=cfg.time_steps,
    )

    # --------------------------------------------------
    # Neuron parameters
    # --------------------------------------------------

    neuron_params = dict(
        tau_m=cfg.tau_m,
        v_rest=cfg.v_rest,
        v_reset=cfg.v_reset,
        v_threshold=cfg.v_threshold,
        tau_adaptation=cfg.tau_adaptation,
        adaptation_strength=cfg.adaptation_strength,
    )

    # --------------------------------------------------
    # STDP parameters
    # --------------------------------------------------

    stdp_params = dict(
        a_plus=cfg.a_plus,
        a_minus=cfg.a_minus,
        tau_plus=cfg.tau_plus,
        tau_minus=cfg.tau_minus,
        w_min=cfg.w_min,
        w_max=cfg.w_max,
    )

    # --------------------------------------------------
    # Homeostasis parameters
    # --------------------------------------------------

    homeostasis_params = dict(
        target_rate=cfg.target_rate,
        tau_homeostasis=cfg.tau_homeostasis,
        strength=cfg.homeostasis_strength,
    )

    # --------------------------------------------------
    # Network
    # --------------------------------------------------

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

    print(model)

    # --------------------------------------------------
    # STDP training
    # --------------------------------------------------

    print("\nStarting STDP...\n")

    model.train()

    history = {
        "layer1_min": [],
        "layer1_mean": [],
        "layer1_max": [],

        "layer2_min": [],
        "layer2_mean": [],
        "layer2_max": [],

        "rate1": [],
        "rate2": [],
    }

    with torch.no_grad():

        for epoch in range(
            cfg.stdp_pretrain_epochs
        ):

            print(
                f"Epoch {epoch + 1}/"
                f"{cfg.stdp_pretrain_epochs}"
            )

            # ------------------------------------------
            # Process all batches
            # ------------------------------------------

            for spikes, _ in train_loader:

                spikes = spikes.permute(
                    1,
                    0,
                    2,
                ).to(device)

                logits, spikes1, spikes2 = model(
                    spikes,
                    apply_stdp=True,
                    return_activity=True,
                )

            # ------------------------------------------
            # Weight statistics
            # ------------------------------------------

            stats1 = get_weight_statistics(
                model.layer1.weights
            )

            stats2 = get_weight_statistics(
                model.layer2.weights
            )

            history["layer1_min"].append(
                stats1["min"]
            )

            history["layer1_mean"].append(
                stats1["mean"]
            )

            history["layer1_max"].append(
                stats1["max"]
            )

            history["layer2_min"].append(
                stats2["min"]
            )

            history["layer2_mean"].append(
                stats2["mean"]
            )

            history["layer2_max"].append(
                stats2["max"]
            )

            # ------------------------------------------
            # Firing rates
            # ------------------------------------------

            rate1 = (
                spikes1
                .float()
                .mean()
                .item()
            )

            rate2 = (
                spikes2
                .float()
                .mean()
                .item()
            )

            history["rate1"].append(rate1)
            history["rate2"].append(rate2)

            # ------------------------------------------
            # Print statistics
            # ------------------------------------------

            print()

            print(
                f"Layer1 weights: "
                f"{stats1['min']:.6f} "
                f"... "
                f"{stats1['mean']:.6f} "
                f"... "
                f"{stats1['max']:.6f}"
            )

            print(
                f"Layer2 weights: "
                f"{stats2['min']:.6f} "
                f"... "
                f"{stats2['mean']:.6f} "
                f"... "
                f"{stats2['max']:.6f}"
            )

            print(
                f"Layer1 firing rate: "
                f"{rate1:.6f}"
            )

            print(
                f"Layer2 firing rate: "
                f"{rate2:.6f}"
            )

    # --------------------------------------------------
    # Analysis
    # --------------------------------------------------

    print("\nRunning analysis...\n")

    analyze_stdp(
        model=model,
        history=history,
        results_dir=run_dir,
    )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    checkpoint_path = os.path.join(
        run_dir,
        "stdp_model.pt",
    )

    torch.save(
        model.state_dict(),
        checkpoint_path,
    )

    # --------------------------------------------------
    # Finished
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("Finished.")
    print("=" * 60)

    print(
        f"Results saved to: {run_dir}"
    )

    print(
        f"Model saved to:   {checkpoint_path}"
    )


if __name__ == "__main__":
    main()