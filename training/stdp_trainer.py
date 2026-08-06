"""
Trainer for unsupervised STDP learning.
"""

import torch

from .metrics import firing_rate


class STDPTrainer:

    def __init__(
        self,
        model,
        train_loader,
        device="cpu",
    ):

        self.device = torch.device(device)

        self.model = model.to(self.device)

        self.train_loader = train_loader

    #################################################################

    @torch.no_grad()
    def stdp_epoch(self):

        self.model.train()

        total_rate1 = 0.0
        total_rate2 = 0.0

        for spikes, _ in self.train_loader:

            spikes = spikes.permute(1, 0, 2).to(self.device)

            layer1, layer2 = self.model(

                spikes,

                apply_stdp=True,

                return_activity=True,

            )

            total_rate1 += firing_rate(layer1)
            total_rate2 += firing_rate(layer2)

        n = len(self.train_loader)

        return {

            "rate1": total_rate1 / n,

            "rate2": total_rate2 / n,

            "layer1_mean": self.model.layer1.weights.mean().item(),

            "layer2_mean": self.model.layer2.weights.mean().item(),

            "layer1_min": self.model.layer1.weights.min().item(),

            "layer1_max": self.model.layer1.weights.max().item(),

            "layer2_min": self.model.layer2.weights.min().item(),

            "layer2_max": self.model.layer2.weights.max().item(),

        }

    #################################################################

    def fit(self, epochs):

        history = {

            "rate1": [],

            "rate2": [],

            "layer1_mean": [],

            "layer2_mean": [],

        }

        for epoch in range(epochs):

            stats = self.stdp_epoch()

            history["rate1"].append(stats["rate1"])
            history["rate2"].append(stats["rate2"])

            history["layer1_mean"].append(stats["layer1_mean"])
            history["layer2_mean"].append(stats["layer2_mean"])

            print()

            print("=" * 60)

            print(f"STDP Epoch {epoch + 1}/{epochs}")

            print(f"Layer1 Rate : {100*stats['rate1']:.2f}%")
            print(f"Layer2 Rate : {100*stats['rate2']:.2f}%")

            print(
                f"Layer1 weights: "
                f"{stats['layer1_min']:.3f} "
                f"... {stats['layer1_mean']:.3f} "
                f"... {stats['layer1_max']:.3f}"
            )

            print(
                f"Layer2 weights: "
                f"{stats['layer2_min']:.3f} "
                f"... {stats['layer2_mean']:.3f} "
                f"... {stats['layer2_max']:.3f}"
            )

            print("=" * 60)

        return history