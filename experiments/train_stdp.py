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
Save learned weights
"""

import torch

from configs import Config

from datasets.dataloader import get_dataloaders

from network.hierarchical_snn import HierarchicalSNN


def main():

    cfg = Config()

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

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
    # Network
    # --------------------------------------------------

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
    ).to(device)

    print(model)

    # --------------------------------------------------
    # STDP training
    # --------------------------------------------------

    print("\nStarting STDP...\n")

    model.train()

    with torch.no_grad():

        for epoch in range(cfg.stdp_pretrain_epochs):

            print(f"Epoch {epoch+1}/{cfg.stdp_pretrain_epochs}")

            for spikes, _ in train_loader:

                spikes = spikes.permute(1, 0, 2).to(device)

                model(
                    spikes,
                    apply_stdp=True,
                )

            print(
                "Layer1:",
                model.layer1.weights.min().item(),
                model.layer1.weights.mean().item(),
                model.layer1.weights.max().item(),
            )

            print(
                "Layer2:",
                model.layer2.weights.min().item(),
                model.layer2.weights.mean().item(),
                model.layer2.weights.max().item(),
            )

    # --------------------------------------------------
    # Save weights
    # --------------------------------------------------

    torch.save(
        model.state_dict(),
        "checkpoints/stdp_model.pt",
    )

    print("\nFinished.")
    print("Weights saved to checkpoints/stdp_model.pt")


if __name__ == "__main__":
    main()