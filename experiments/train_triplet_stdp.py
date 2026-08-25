"""
Pure Triplet STDP training with LTP/LTD diagnostics.

Pipeline:

SHD
 ↓
Input spikes
 ↓
TripletHierarchicalSNN
 ↓
Triplet STDP
 ↓
LTP / LTD diagnostics
 ↓
Weight statistics
 ↓
Firing rates
 ↓
Save results

This experiment is completely separate from Pair-STDP.
"""

import os
import torch

from configs import Config

from datasets.dataloader import get_dataloaders

from network.triplet.triplet_snn import TripletHierarchicalSNN

from utils.experiment import (
    create_run_directory,
    save_config,
    save_json,
)


# ============================================================
# Helpers
# ============================================================

def get_weight_statistics(weights):
    """
    Return basic statistics for a weight matrix.
    """

    weights = weights.detach()

    return {
        "min": weights.min().item(),
        "mean": weights.mean().item(),
        "max": weights.max().item(),
        "std": weights.std().item(),
        "median": weights.median().item(),
    }


def get_delta_statistics(delta):
    """
    Return statistics for a weight update.
    """

    delta = delta.detach()

    total = delta.numel()

    increased = (
        delta > 1e-12
    ).sum().item()

    decreased = (
        delta < -1e-12
    ).sum().item()

    near_zero = (
        delta.abs() <= 1e-12
    ).sum().item()

    return {
        "min": delta.min().item(),
        "mean": delta.mean().item(),
        "max": delta.max().item(),
        "std": delta.std().item(),
        "median": delta.median().item(),

        "increased_fraction": (
            increased / total
        ),

        "decreased_fraction": (
            decreased / total
        ),

        "near_zero_fraction": (
            near_zero / total
        ),
    }


def get_activity_statistics(spikes):
    """
    Calculate firing-rate statistics.
    """

    spikes = spikes.float()

    return {
        "mean_rate": spikes.mean().item(),
        "total_spikes": spikes.sum().item(),
    }


def get_plasticity_statistics(
    ltp,
    ltd,
    delta,
):
    """
    Calculate LTP/LTD/net statistics.
    """

    ltp = ltp.detach()
    ltd = ltd.detach()
    delta = delta.detach()

    return {
        "ltp_mean": ltp.mean().item(),
        "ltp_std": ltp.std().item(),
        "ltp_min": ltp.min().item(),
        "ltp_max": ltp.max().item(),

        "ltd_mean": ltd.mean().item(),
        "ltd_std": ltd.std().item(),
        "ltd_min": ltd.min().item(),
        "ltd_max": ltd.max().item(),

        "net_mean": delta.mean().item(),
        "net_std": delta.std().item(),
        "net_min": delta.min().item(),
        "net_max": delta.max().item(),

        "total_ltp": ltp.sum().item(),
        "total_ltd": ltd.sum().item(),
        "total_net": delta.sum().item(),
    }


def print_plasticity_statistics(
    name,
    stats,
):
    """
    Pretty-print LTP/LTD statistics.
    """

    print()
    print("=" * 60)
    print(f"{name} PLASTICITY")
    print("=" * 60)

    print()
    print("LTP")
    print(
        f"  mean  : {stats['ltp_mean']:+.10e}"
    )
    print(
        f"  std   : {stats['ltp_std']:.10e}"
    )
    print(
        f"  min   : {stats['ltp_min']:+.10e}"
    )
    print(
        f"  max   : {stats['ltp_max']:+.10e}"
    )

    print()
    print("LTD")
    print(
        f"  mean  : {stats['ltd_mean']:+.10e}"
    )
    print(
        f"  std   : {stats['ltd_std']:.10e}"
    )
    print(
        f"  min   : {stats['ltd_min']:+.10e}"
    )
    print(
        f"  max   : {stats['ltd_max']:+.10e}"
    )

    print()
    print("NET ΔW")
    print(
        f"  mean  : {stats['net_mean']:+.10e}"
    )
    print(
        f"  std   : {stats['net_std']:.10e}"
    )
    print(
        f"  min   : {stats['net_min']:+.10e}"
    )
    print(
        f"  max   : {stats['net_max']:+.10e}"
    )

    print()
    print("TOTAL")
    print(
        f"  LTP   : {stats['total_ltp']:+.10e}"
    )
    print(
        f"  LTD   : {stats['total_ltd']:+.10e}"
    )
    print(
        f"  NET   : {stats['total_net']:+.10e}"
    )


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Configuration
    # ========================================================

    cfg = Config()

    # Keep Triplet-STDP results completely separate.
    run_dir = create_run_directory(
        base="results",
        experiment="triplet_stdp",
    )

    save_config(
        cfg,
        os.path.join(
            run_dir,
            "config.json",
        ),
    )

    print("=" * 60)
    print("Pure Triplet STDP training")
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

    # IMPORTANT:
    # Limit the number of batches for diagnostics.
    #
    # This keeps the experiment fast.
    #
    # Change to None when we are ready for the full run.
    batch_limit = 5

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
    # Triplet STDP parameters
    # ========================================================

    triplet_params = dict(

        # Pair terms
        a2_plus=cfg.a_plus,
        a2_minus=cfg.a_minus,

        # Triplet terms
        a3_plus=cfg.a3_plus,
        a3_minus=cfg.a3_minus,

        # Fast traces
        tau_plus=cfg.tau_plus,
        tau_minus=cfg.tau_minus,

        # Slow traces
        tau_x=cfg.tau_pre_slow,
        tau_y=cfg.tau_post_slow,

        # Weight bounds
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

    model = TripletHierarchicalSNN(
        n_input=cfg.n_input,
        n_hidden1=cfg.hidden1,
        n_hidden2=cfg.hidden2,
        n_classes=cfg.n_classes,

        neuron_params=neuron_params,

        triplet_params1=triplet_params,
        triplet_params2=triplet_params,

        homeostasis_params=homeostasis_params,

        input_gain=cfg.input_gain,

        use_classifier=False,
    ).to(device)

    print()
    print("Network created successfully.")

    # ========================================================
    # Initial weights
    # ========================================================

    initial_weights_layer1 = (
        model.layer1.weights
        .detach()
        .cpu()
        .clone()
    )

    initial_weights_layer2 = (
        model.layer2.weights
        .detach()
        .cpu()
        .clone()
    )

    # Save initial matrices
    torch.save(
        initial_weights_layer1,
        os.path.join(
            run_dir,
            "initial_weights_layer1.pt",
        ),
    )

    torch.save(
        initial_weights_layer2,
        os.path.join(
            run_dir,
            "initial_weights_layer2.pt",
        ),
    )

    initial_stats = {
        "layer1": get_weight_statistics(
            initial_weights_layer1
        ),
        "layer2": get_weight_statistics(
            initial_weights_layer2
        ),
    }

    save_json(
        initial_stats,
        os.path.join(
            run_dir,
            "initial_weight_statistics.json",
        ),
    )

    # ========================================================
    # History
    # ========================================================

    history = {
        "epoch": [],

        "layer1_min": [],
        "layer1_mean": [],
        "layer1_max": [],
        "layer1_std": [],

        "layer2_min": [],
        "layer2_mean": [],
        "layer2_max": [],
        "layer2_std": [],

        "rate1": [],
        "rate2": [],

        "ltp1_mean": [],
        "ltd1_mean": [],
        "delta1_mean": [],

        "ltp2_mean": [],
        "ltd2_mean": [],
        "delta2_mean": [],
    }

    # ========================================================
    # Training
    # ========================================================

    print()
    print("Starting Triplet STDP...")
    print()

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

            # -----------------------------------------------
            # Process batches
            # -----------------------------------------------

            for batch_idx, (spikes, _) in enumerate(
                train_loader
            ):

                if (
                    batch_limit is not None
                    and batch_idx >= batch_limit
                ):
                    break

                print(
                    f"Batch {batch_idx + 1}/"
                    f"{batch_limit}"
                )

                # [batch, time, input]
                #
                # ↓
                #
                # [time, batch, input]

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

            # -----------------------------------------------
            # Weight statistics
            # -----------------------------------------------

            stats1 = get_weight_statistics(
                model.layer1.weights
            )

            stats2 = get_weight_statistics(
                model.layer2.weights
            )

            # -----------------------------------------------
            # Firing rates
            # -----------------------------------------------

            activity1 = get_activity_statistics(
                last_spikes1
            )

            activity2 = get_activity_statistics(
                last_spikes2
            )

            rate1 = activity1["mean_rate"]
            rate2 = activity2["mean_rate"]

            # -----------------------------------------------
            # Plasticity diagnostics
            #
            # These correspond to the LAST processed batch.
            # -----------------------------------------------

            ltp1 = model.layer1.last_ltp
            ltd1 = model.layer1.last_ltd
            delta1 = model.layer1.last_delta_w

            ltp2 = model.layer2.last_ltp
            ltd2 = model.layer2.last_ltd
            delta2 = model.layer2.last_delta_w

            plasticity1 = get_plasticity_statistics(
                ltp1,
                ltd1,
                delta1,
            )

            plasticity2 = get_plasticity_statistics(
                ltp2,
                ltd2,
                delta2,
            )

            # -----------------------------------------------
            # History
            # -----------------------------------------------

            history["epoch"].append(
                epoch + 1
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

            history["layer1_std"].append(
                stats1["std"]
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

            history["layer2_std"].append(
                stats2["std"]
            )

            history["rate1"].append(
                rate1
            )

            history["rate2"].append(
                rate2
            )

            history["ltp1_mean"].append(
                plasticity1["ltp_mean"]
            )

            history["ltd1_mean"].append(
                plasticity1["ltd_mean"]
            )

            history["delta1_mean"].append(
                plasticity1["net_mean"]
            )

            history["ltp2_mean"].append(
                plasticity2["ltp_mean"]
            )

            history["ltd2_mean"].append(
                plasticity2["ltd_mean"]
            )

            history["delta2_mean"].append(
                plasticity2["net_mean"]
            )

            # -----------------------------------------------
            # Print weight statistics
            # -----------------------------------------------

            print()

            print(
                f"Layer 1 weights: "
                f"{stats1['min']:.6f}"
                f" ... "
                f"{stats1['mean']:.6f}"
                f" ... "
                f"{stats1['max']:.6f}"
            )

            print(
                f"Layer 2 weights: "
                f"{stats2['min']:.6f}"
                f" ... "
                f"{stats2['mean']:.6f}"
                f" ... "
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

            # -----------------------------------------------
            # Print plasticity
            # -----------------------------------------------

            print_plasticity_statistics(
                "LAYER 1",
                plasticity1,
            )

            print_plasticity_statistics(
                "LAYER 2",
                plasticity2,
            )

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
    # Final weights
    # ========================================================

    final_weights_layer1 = (
        model.layer1.weights
        .detach()
        .cpu()
        .clone()
    )

    final_weights_layer2 = (
        model.layer2.weights
        .detach()
        .cpu()
        .clone()
    )

    # Save final matrices

    torch.save(
        final_weights_layer1,
        os.path.join(
            run_dir,
            "final_weights_layer1.pt",
        ),
    )

    torch.save(
        final_weights_layer2,
        os.path.join(
            run_dir,
            "final_weights_layer2.pt",
        ),
    )

    # ========================================================
    # Weight changes
    # ========================================================

    delta_weights_layer1 = (
        final_weights_layer1
        - initial_weights_layer1
    )

    delta_weights_layer2 = (
        final_weights_layer2
        - initial_weights_layer2
    )

    torch.save(
        delta_weights_layer1,
        os.path.join(
            run_dir,
            "delta_weights_layer1.pt",
        ),
    )

    torch.save(
        delta_weights_layer2,
        os.path.join(
            run_dir,
            "delta_weights_layer2.pt",
        ),
    )

    # ========================================================
    # Final weight statistics
    # ========================================================

    final_stats = {
        "layer1": get_weight_statistics(
            final_weights_layer1
        ),
        "layer2": get_weight_statistics(
            final_weights_layer2
        ),
    }

    save_json(
        final_stats,
        os.path.join(
            run_dir,
            "final_weight_statistics.json",
        ),
    )

    # ========================================================
    # Delta statistics
    # ========================================================

    delta_stats = {
        "layer1": get_delta_statistics(
            delta_weights_layer1
        ),
        "layer2": get_delta_statistics(
            delta_weights_layer2
        ),
    }

    save_json(
        delta_stats,
        os.path.join(
            run_dir,
            "delta_weight_statistics.json",
        ),
    )

    # ========================================================
    # Save final plasticity diagnostics
    # ========================================================

    final_plasticity = {
        "layer1": plasticity1,
        "layer2": plasticity2,
    }

    save_json(
        final_plasticity,
        os.path.join(
            run_dir,
            "final_plasticity_statistics.json",
        ),
    )

    # Save matrices

    torch.save(
        ltp1.cpu(),
        os.path.join(
            run_dir,
            "ltp_layer1.pt",
        ),
    )

    torch.save(
        ltd1.cpu(),
        os.path.join(
            run_dir,
            "ltd_layer1.pt",
        ),
    )

    torch.save(
        delta1.cpu(),
        os.path.join(
            run_dir,
            "stdp_delta_layer1.pt",
        ),
    )

    torch.save(
        ltp2.cpu(),
        os.path.join(
            run_dir,
            "ltp_layer2.pt",
        ),
    )

    torch.save(
        ltd2.cpu(),
        os.path.join(
            run_dir,
            "ltd_layer2.pt",
        ),
    )

    torch.save(
        delta2.cpu(),
        os.path.join(
            run_dir,
            "stdp_delta_layer2.pt",
        ),
    )

    # ========================================================
    # Save model
    # ========================================================

    checkpoint_path = os.path.join(
        run_dir,
        "triplet_stdp_model.pt",
    )

    torch.save(
        model.state_dict(),
        checkpoint_path,
    )

    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "experiment": "triplet_stdp",

        "device": str(device),

        "epochs": cfg.stdp_pretrain_epochs,

        "time_steps": cfg.time_steps,

        "batch_limit": batch_limit,

        "batches_available": len(
            train_loader
        ),

        "final_layer1_rate": history[
            "rate1"
        ][-1],

        "final_layer2_rate": history[
            "rate2"
        ][-1],

        "layer1_ltp_mean": plasticity1[
            "ltp_mean"
        ],

        "layer1_ltd_mean": plasticity1[
            "ltd_mean"
        ],

        "layer1_net_mean": plasticity1[
            "net_mean"
        ],

        "layer2_ltp_mean": plasticity2[
            "ltp_mean"
        ],

        "layer2_ltd_mean": plasticity2[
            "ltd_mean"
        ],

        "layer2_net_mean": plasticity2[
            "net_mean"
        ],

        "checkpoint": checkpoint_path,
    }

    save_json(
        summary,
        os.path.join(
            run_dir,
            "summary.json",
        ),
    )

    # ========================================================
    # Finished
    # ========================================================

    print()
    print("=" * 60)
    print("Triplet STDP training finished.")
    print("=" * 60)

    print(
        f"Results saved to: {run_dir}"
    )

    print(
        f"Model saved to:   {checkpoint_path}"
    )

    print()
    print("Saved diagnostics:")

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

    print(
        "  final_plasticity_statistics.json"
    )

    print(
        "  delta_weight_statistics.json"
    )


if __name__ == "__main__":
    main()