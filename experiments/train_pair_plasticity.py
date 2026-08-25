"""
Pure Pair STDP plasticity experiment.

Pipeline

SHD
 |
 v
HierarchicalSNN
 |
 +--> Pair STDP Layer 1
 |
 +--> Pair STDP Layer 2
 |
 v
Plasticity diagnostics

Saves:
- initial/final weights
- delta weights
- LTP/LTD matrices
- plasticity statistics
- firing rates
- experiment summary

IMPORTANT:
All results are stored separately from Triplet STDP.
"""

import os
import torch

from configs import Config

from datasets.dataloader import get_dataloaders

from network.hierarchical_snn import HierarchicalSNN

from utils.experiment import (
    create_run_directory,
    save_config,
    save_json,
)


# ============================================================
# Statistics
# ============================================================

def get_statistics(tensor):
    """
    Basic tensor statistics.
    """

    return {
        "min": tensor.min().item(),
        "mean": tensor.mean().item(),
        "max": tensor.max().item(),
        "std": tensor.std().item(),
        "median": tensor.median().item(),
    }


def get_plasticity_statistics(delta):
    """
    Statistics describing the direction and magnitude
    of synaptic changes.
    """

    near_zero_threshold = 1e-8

    increased = (
        delta > near_zero_threshold
    ).float().mean().item()

    decreased = (
        delta < -near_zero_threshold
    ).float().mean().item()

    near_zero = (
        delta.abs() <= near_zero_threshold
    ).float().mean().item()

    return {
        "min": delta.min().item(),
        "mean": delta.mean().item(),
        "max": delta.max().item(),
        "std": delta.std().item(),
        "median": delta.median().item(),

        "increased_fraction": increased,
        "decreased_fraction": decreased,
        "near_zero_fraction": near_zero,

        "total": delta.sum().item(),
    }


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Configuration
    # ========================================================

    cfg = Config()

    run_dir = create_run_directory(
        base="results",
        experiment="pair_stdp",
    )

    save_config(
        cfg,
        os.path.join(
            run_dir,
            "config.json",
        ),
    )

    print("=" * 60)
    print("PURE PAIR STDP PLASTICITY")
    print("=" * 60)

    print(
        f"Results directory: {run_dir}"
    )

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        cfg.device
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device     : {device}"
    )

    print(
        f"Epochs     : {cfg.stdp_pretrain_epochs}"
    )

    print(
        f"Time steps : {cfg.time_steps}"
    )

    print("=" * 60)

    # ========================================================
    # Dataset
    # ========================================================

    train_loader, _ = get_dataloaders(
        root=cfg.dataset_root,
        batch_size=cfg.batch_size,
        time_steps=cfg.time_steps,
    )

    print(
        f"Batches available: {len(train_loader)}"
    )

    # ========================================================
    # Limit number of batches
    #
    # We use the same short experiment as Triplet STDP.
    # This makes Pair and Triplet directly comparable.
    # ========================================================

    batch_limit = getattr(
        cfg,
        "stdp_batch_limit",
        5,
    )

    print(
        f"Batch limit      : {batch_limit}"
    )

    # ========================================================
    # Neuron parameters
    # ========================================================

    neuron_params = dict(
        tau_m=cfg.tau_m,
        v_rest=cfg.v_rest,
        v_reset=cfg.v_reset,
        v_threshold=cfg.v_threshold,
        tau_adaptation=cfg.tau_adaptation,
        adaptation_strength=cfg.adaptation_strength,
    )

    # ========================================================
    # Pair STDP parameters
    # ========================================================

    stdp_params = dict(
        a_plus=cfg.a_plus,
        a_minus=cfg.a_minus,
        tau_plus=cfg.tau_plus,
        tau_minus=cfg.tau_minus,
        w_min=cfg.w_min,
        w_max=cfg.w_max,
    )

    # ========================================================
    # Homeostasis
    # ========================================================

    homeostasis_params = dict(
        target_rate=cfg.target_rate,
        tau_homeostasis=cfg.tau_homeostasis,
        strength=cfg.homeostasis_strength,
    )

    # ========================================================
    # Network
    # ========================================================

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

        use_classifier=False,
    ).to(device)

    print(
        "\nNetwork created successfully."
    )

    # ========================================================
    # Initial weights
    # ========================================================

    initial_layer1 = (
        model.layer1.weights
        .detach()
        .cpu()
        .clone()
    )

    initial_layer2 = (
        model.layer2.weights
        .detach()
        .cpu()
        .clone()
    )

    torch.save(
        initial_layer1,
        os.path.join(
            run_dir,
            "initial_weights_layer1.pt",
        ),
    )

    torch.save(
        initial_layer2,
        os.path.join(
            run_dir,
            "initial_weights_layer2.pt",
        ),
    )

    # ========================================================
    # Training
    # ========================================================

    history = {
        "epoch": [],

        "layer1_mean": [],
        "layer2_mean": [],

        "layer1_rate": [],
        "layer2_rate": [],
    }

    print(
        "\nStarting Pair STDP...\n"
    )

    model.train()

    with torch.no_grad():

        for epoch in range(
            cfg.stdp_pretrain_epochs
        ):

            print(
                f"Epoch {epoch + 1}/"
                f"{cfg.stdp_pretrain_epochs}"
            )

            last_spikes1 = None
            last_spikes2 = None

            actual_batches = min(
                batch_limit,
                len(train_loader),
            )

            for batch_idx, (
                spikes,
                _
            ) in enumerate(train_loader):

                if batch_idx >= batch_limit:
                    break

                print(
                    f"Batch {batch_idx + 1}/"
                    f"{actual_batches}"
                )

                spikes = spikes.permute(
                    1,
                    0,
                    2,
                ).to(device)

                spikes1, spikes2 = model(
                    spikes,
                    apply_stdp=True,
                    return_activity=True,
                )

                last_spikes1 = spikes1
                last_spikes2 = spikes2

            # ------------------------------------------------
            # Weight statistics
            # ------------------------------------------------

            stats1 = get_statistics(
                model.layer1.weights
            )

            stats2 = get_statistics(
                model.layer2.weights
            )

            # ------------------------------------------------
            # Firing rates
            # ------------------------------------------------

            rate1 = (
                last_spikes1
                .float()
                .mean()
                .item()
            )

            rate2 = (
                last_spikes2
                .float()
                .mean()
                .item()
            )

            history["epoch"].append(
                epoch + 1
            )

            history["layer1_mean"].append(
                stats1["mean"]
            )

            history["layer2_mean"].append(
                stats2["mean"]
            )

            history["layer1_rate"].append(
                rate1
            )

            history["layer2_rate"].append(
                rate2
            )

            print()

            print(
                f"Layer 1 weights: "
                f"{stats1['min']:.6f} ... "
                f"{stats1['mean']:.6f} ... "
                f"{stats1['max']:.6f}"
            )

            print(
                f"Layer 2 weights: "
                f"{stats2['min']:.6f} ... "
                f"{stats2['mean']:.6f} ... "
                f"{stats2['max']:.6f}"
            )

            print(
                f"Layer 1 firing rate: "
                f"{rate1:.6%}"
            )

            print(
                f"Layer 2 firing rate: "
                f"{rate2:.6%}"
            )

    # ========================================================
    # Final weights
    # ========================================================

    final_layer1 = (
        model.layer1.weights
        .detach()
        .cpu()
        .clone()
    )

    final_layer2 = (
        model.layer2.weights
        .detach()
        .cpu()
        .clone()
    )

    # ========================================================
    # Delta weights
    # ========================================================

    delta_layer1 = (
        final_layer1
        - initial_layer1
    )

    delta_layer2 = (
        final_layer2
        - initial_layer2
    )

    # ========================================================
    # Save weight matrices
    # ========================================================

    torch.save(
        final_layer1,
        os.path.join(
            run_dir,
            "final_weights_layer1.pt",
        ),
    )

    torch.save(
        final_layer2,
        os.path.join(
            run_dir,
            "final_weights_layer2.pt",
        ),
    )

    torch.save(
        delta_layer1,
        os.path.join(
            run_dir,
            "delta_weights_layer1.pt",
        ),
    )

    torch.save(
        delta_layer2,
        os.path.join(
            run_dir,
            "delta_weights_layer2.pt",
        ),
    )

    # ========================================================
    # Separate LTP / LTD
    #
    # Positive delta = LTP
    # Negative delta = LTD magnitude
    # ========================================================

    ltp_layer1 = torch.clamp(
        delta_layer1,
        min=0.0,
    )

    ltd_layer1 = torch.clamp(
        -delta_layer1,
        min=0.0,
    )

    ltp_layer2 = torch.clamp(
        delta_layer2,
        min=0.0,
    )

    ltd_layer2 = torch.clamp(
        -delta_layer2,
        min=0.0,
    )

    torch.save(
        ltp_layer1,
        os.path.join(
            run_dir,
            "ltp_layer1.pt",
        ),
    )

    torch.save(
        ltd_layer1,
        os.path.join(
            run_dir,
            "ltd_layer1.pt",
        ),
    )

    torch.save(
        ltp_layer2,
        os.path.join(
            run_dir,
            "ltp_layer2.pt",
        ),
    )

    torch.save(
        ltd_layer2,
        os.path.join(
            run_dir,
            "ltd_layer2.pt",
        ),
    )

    # ========================================================
    # Plasticity statistics
    # ========================================================

    plasticity_stats = {

        "layer1": {
            "ltp": get_statistics(
                ltp_layer1
            ),
            "ltd": get_statistics(
                ltd_layer1
            ),
            "net": get_plasticity_statistics(
                delta_layer1
            ),
        },

        "layer2": {
            "ltp": get_statistics(
                ltp_layer2
            ),
            "ltd": get_statistics(
                ltd_layer2
            ),
            "net": get_plasticity_statistics(
                delta_layer2
            ),
        },
    }

    save_json(
        plasticity_stats,
        os.path.join(
            run_dir,
            "plasticity_statistics.json",
        ),
    )

    # ========================================================
    # Weight statistics
    # ========================================================

    weight_statistics = {

        "initial": {
            "layer1": get_statistics(
                initial_layer1
            ),
            "layer2": get_statistics(
                initial_layer2
            ),
        },

        "final": {
            "layer1": get_statistics(
                final_layer1
            ),
            "layer2": get_statistics(
                final_layer2
            ),
        },
    }

    save_json(
        weight_statistics,
        os.path.join(
            run_dir,
            "weight_statistics.json",
        ),
    )

    # ========================================================
    # History
    # ========================================================

    save_json(
        history,
        os.path.join(
            run_dir,
            "history.json",
        ),
    )

    # ========================================================
    # Model
    # ========================================================

    checkpoint_path = os.path.join(
        run_dir,
        "pair_stdp_model.pt",
    )

    torch.save(
        model.state_dict(),
        checkpoint_path,
    )

    # ========================================================
    # Summary
    # ========================================================

    summary = {

        "experiment": "pair_stdp",

        "device": str(device),

        "epochs":
            cfg.stdp_pretrain_epochs,

        "time_steps":
            cfg.time_steps,

        "batches_available":
            len(train_loader),

        "batches_used":
            actual_batches,

        "final_layer1_rate":
            history["layer1_rate"][-1],

        "final_layer2_rate":
            history["layer2_rate"][-1],

        "checkpoint":
            checkpoint_path,
    }

    save_json(
        summary,
        os.path.join(
            run_dir,
            "summary.json",
        ),
    )

    # ========================================================
    # Print final plasticity
    # ========================================================

    print()
    print("=" * 60)
    print("PAIR STDP PLASTICITY")
    print("=" * 60)

    for name, delta in (
        ("Layer 1", delta_layer1),
        ("Layer 2", delta_layer2),
    ):

        stats = get_plasticity_statistics(
            delta
        )

        print()
        print(name)

        print(
            f"  ΔW mean   : "
            f"{stats['mean']:+.10e}"
        )

        print(
            f"  ΔW std    : "
            f"{stats['std']:.10e}"
        )

        print(
            f"  ΔW min    : "
            f"{stats['min']:+.10e}"
        )

        print(
            f"  ΔW max    : "
            f"{stats['max']:+.10e}"
        )

        print(
            f"  Increased : "
            f"{stats['increased_fraction']:.4%}"
        )

        print(
            f"  Decreased : "
            f"{stats['decreased_fraction']:.4%}"
        )

        print(
            f"  Near zero : "
            f"{stats['near_zero_fraction']:.4%}"
        )

    print()
    print("=" * 60)
    print("Pair STDP training finished.")
    print("=" * 60)

    print(
        f"Results saved to: {run_dir}"
    )

    print(
        f"Model saved to:   {checkpoint_path}"
    )

    print()
    print("Saved matrices:")

    print(
        "  initial_weights_layer1.pt"
    )

    print(
        "  initial_weights_layer2.pt"
    )

    print(
        "  final_weights_layer1.pt"
    )

    print(
        "  final_weights_layer2.pt"
    )

    print(
        "  delta_weights_layer1.pt"
    )

    print(
        "  delta_weights_layer2.pt"
    )

    print(
        "  ltp_layer1.pt"
    )

    print(
        "  ltd_layer1.pt"
    )

    print(
        "  ltp_layer2.pt"
    )

    print(
        "  ltd_layer2.pt"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()