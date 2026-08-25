"""
Smoke test for the complete Triplet STDP pipeline.

Checks:

1. SHD batch loads correctly.
2. TripletHierarchicalSNN accepts the data.
3. Layer 1 and Layer 2 produce spikes.
4. Triplet STDP changes the weights.
5. Weight shapes remain correct.

This test does NOT modify Pair STDP code.
"""

import torch

from configs import Config
from datasets.dataloader import get_dataloaders
from network.triplet.triplet_snn import TripletHierarchicalSNN


def main():

    print("=" * 60)
    print("TRIPLET STDP PIPELINE SMOKE TEST")
    print("=" * 60)

    # ==================================================
    # Configuration
    # ==================================================

    cfg = Config()

    device = torch.device(
        cfg.device
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # ==================================================
    # Dataset
    # ==================================================

    train_loader, _ = get_dataloaders(
        root=cfg.dataset_root,
        batch_size=cfg.batch_size,
        time_steps=cfg.time_steps,
    )

    spikes, labels = next(iter(train_loader))

    print()
    print("Input batch")
    print("-----------")
    print("Shape :", tuple(spikes.shape))
    print("Labels:", tuple(labels.shape))

    # Expected:
    #
    # [batch, time, input]
    #
    # e.g.
    # [32, 200, 700]

    spikes = spikes.permute(
        1,
        0,
        2,
    ).to(device)

    print(
        "Network input:",
        tuple(spikes.shape),
    )

    # ==================================================
    # Neuron parameters
    # ==================================================

    neuron_params = dict(
        tau_m=cfg.tau_m,
        v_rest=cfg.v_rest,
        v_reset=cfg.v_reset,
        v_threshold=cfg.v_threshold,
        tau_adaptation=cfg.tau_adaptation,
        adaptation_strength=cfg.adaptation_strength,
    )

    # ==================================================
    # Triplet STDP parameters
    # ==================================================

    triplet_params = dict(

        # Pair contribution
        a2_plus=cfg.a_plus,
        a2_minus=cfg.a_minus,

        # Triplet contribution
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

    # ==================================================
    # Homeostasis
    # ==================================================

    homeostasis_params = dict(
        target_rate=cfg.target_rate,
        tau_homeostasis=cfg.tau_homeostasis,
        strength=cfg.homeostasis_strength,
    )

    # ==================================================
    # Network
    # ==================================================

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

    # ==================================================
    # Initial weights
    # ==================================================

    weights1_before = (
        model.layer1.weights
        .detach()
        .clone()
    )

    weights2_before = (
        model.layer2.weights
        .detach()
        .clone()
    )

    print()
    print("Initial weights")
    print("---------------")

    print(
        "Layer 1:",
        weights1_before.min().item(),
        weights1_before.mean().item(),
        weights1_before.max().item(),
    )

    print(
        "Layer 2:",
        weights2_before.min().item(),
        weights2_before.mean().item(),
        weights2_before.max().item(),
    )

    # ==================================================
    # One STDP update
    # ==================================================

    print()
    print("Running one Triplet STDP batch...")
    print()

    model.train()

    with torch.no_grad():

        spikes1, spikes2 = model(
            spikes,
            apply_stdp=True,
            return_activity=True,
        )

    # ==================================================
    # Activity
    # ==================================================

    print()
    print("Activity")
    print("--------")

    print(
        "Layer 1 spikes:",
        tuple(spikes1.shape),
    )

    print(
        "Layer 2 spikes:",
        tuple(spikes2.shape),
    )

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

    print(
        f"Layer 1 firing rate: {rate1:.6%}"
    )

    print(
        f"Layer 2 firing rate: {rate2:.6%}"
    )

    # ==================================================
    # Updated weights
    # ==================================================

    weights1_after = (
        model.layer1.weights
        .detach()
    )

    weights2_after = (
        model.layer2.weights
        .detach()
    )

    difference1 = (
        weights1_after - weights1_before
    ).abs()

    difference2 = (
        weights2_after - weights2_before
    ).abs()

    max_diff1 = difference1.max().item()
    mean_diff1 = difference1.mean().item()

    max_diff2 = difference2.max().item()
    mean_diff2 = difference2.mean().item()

    print()
    print("Weight changes")
    print("--------------")

    print(
        f"Layer 1 max difference : "
        f"{max_diff1:.12e}"
    )

    print(
        f"Layer 1 mean difference: "
        f"{mean_diff1:.12e}"
    )

    print(
        f"Layer 2 max difference : "
        f"{max_diff2:.12e}"
    )

    print(
        f"Layer 2 mean difference: "
        f"{mean_diff2:.12e}"
    )

    # ==================================================
    # Shape checks
    # ==================================================

    print()
    print("Shape checks")
    print("------------")

    assert (
        weights1_before.shape
        == (
            cfg.n_input,
            cfg.hidden1,
        )
    )

    assert (
        weights2_before.shape
        == (
            cfg.hidden1,
            cfg.hidden2,
        )
    )

    assert (
        spikes1.shape
        == (
            cfg.time_steps,
            cfg.batch_size,
            cfg.hidden1,
        )
    )

    assert (
        spikes2.shape
        == (
            cfg.time_steps,
            cfg.batch_size,
            cfg.hidden2,
        )
    )

    print("✓ Weight shapes correct")
    print("✓ Spike shapes correct")

    # ==================================================
    # Weight update checks
    # ==================================================

    assert max_diff1 > 0, (
        "Layer 1 weights did not change."
    )

    assert max_diff2 > 0, (
        "Layer 2 weights did not change."
    )

    print("✓ Layer 1 weights changed")
    print("✓ Layer 2 weights changed")

    # ==================================================
    # Bounds
    # ==================================================

    assert (
        weights1_after.min().item()
        >= cfg.w_min
    )

    assert (
        weights1_after.max().item()
        <= cfg.w_max
    )

    assert (
        weights2_after.min().item()
        >= cfg.w_min
    )

    assert (
        weights2_after.max().item()
        <= cfg.w_max
    )

    print("✓ Layer 1 weights within bounds")
    print("✓ Layer 2 weights within bounds")

    # ==================================================
    # Finished
    # ==================================================

    print()
    print("=" * 60)
    print("✓ TRIPLET STDP PIPELINE TEST PASSED")
    print("=" * 60)

    print()
    print("The complete Triplet pipeline works:")
    print()
    print("SHD")
    print(" ↓")
    print("TripletHierarchicalSNN")
    print(" ↓")
    print("TripletSpikingLinear")
    print(" ↓")
    print("TripletSTDP")
    print(" ↓")
    print("Updated weights")


if __name__ == "__main__":
    main()