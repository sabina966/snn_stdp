"""
Trainer for supervised SNN learning.
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from .metrics import accuracy, firing_rate


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        optimizer,
        device="cpu",
    ):

        self.device = torch.device(device)

        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.optimizer = optimizer

        self.criterion = nn.CrossEntropyLoss()

    #################################################################

    def train_epoch(self):

        self.model.train()

        total_loss = 0.0
        total_acc = 0.0

        total_rate1 = 0.0
        total_rate2 = 0.0

        for spikes, labels in tqdm(self.train_loader):

            spikes = spikes.permute(1, 0, 2).to(self.device)
            labels = labels.long().to(self.device)

            self.optimizer.zero_grad()

            logits, layer1, layer2 = self.model(
                spikes,
                return_activity=True,
            )

            loss = self.criterion(
                logits,
                labels,
            )

            loss.backward()

            self.optimizer.step()

            with torch.no_grad():
                self.model.layer1.weights.clamp_(0.0, 1.0)
                self.model.layer2.weights.clamp_(0.0, 1.0)

            print(
                "After optimizer:",
                self.model.layer1.weights.min().item(),
                self.model.layer1.weights.mean().item(),
                self.model.layer1.weights.max().item()
            )

            print(
                "After optimizer L2:",
                self.model.layer2.weights.min().item(),
                self.model.layer2.weights.mean().item(),
                self.model.layer2.weights.max().item()
            )

            total_loss += loss.item()

            total_acc += accuracy(
                logits.detach(),
                labels,
            )

            total_rate1 += firing_rate(layer1)

            total_rate2 += firing_rate(layer2)

        n = len(self.train_loader)

        return {
            "loss": total_loss / n,
            "accuracy": total_acc / n,
            "rate1": total_rate1 / n,
            "rate2": total_rate2 / n,
        }

    #################################################################

    #################################################################

    @torch.no_grad()
    def stdp_pretrain_epoch(self):

        self.model.train()

        for spikes, _ in self.train_loader:

            spikes = spikes.permute(1, 0, 2).to(self.device)

            self.model(
                spikes,
                apply_stdp=True,
            )

        print("\nLayer 1:")
        print(
            f"min={self.model.layer1.weights.min().item():.4f}, "
            f"mean={self.model.layer1.weights.mean().item():.4f}, "
            f"max={self.model.layer1.weights.max().item():.4f}"
        )

        print("Layer 2:")
        print(
            f"min={self.model.layer2.weights.min().item():.4f}, "
            f"mean={self.model.layer2.weights.mean().item():.4f}, "
            f"max={self.model.layer2.weights.max().item():.4f}"
        )

#################################################################

    @torch.no_grad()
    def evaluate(self):

        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0

        all_labels = []
        all_predictions = []

        for spikes, labels in self.test_loader:

            spikes = spikes.permute(1, 0, 2).to(self.device)
            labels = labels.long().to(self.device)

            logits = self.model(spikes)
            prediction = logits.argmax(dim=1)

            all_predictions.append(
                prediction.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

            loss = self.criterion(
                logits,
                labels,
            )

            total_loss += loss.item()

            total_acc += accuracy(
                logits,
                labels,
            )

        n = len(self.test_loader)

        all_predictions = torch.cat(all_predictions)

        all_labels = torch.cat(all_labels)

        return {
            "loss": total_loss / n,
            "accuracy": total_acc / n,
            "labels": all_labels,
            "predictions": all_predictions,
        }

    #################################################################

    def fit(self, epochs):

        history = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
            "rate1": [],
            "rate2": [],
        }

        best_test_accuracy = -1.0
        best_epoch = 0
        best_state_dict = None
        best_labels = None
        best_predictions = None

        for epoch in range(epochs):

            train = self.train_epoch()

            test = self.evaluate()

            history["train_loss"].append(train["loss"])
            history["train_acc"].append(train["accuracy"])

            history["test_loss"].append(test["loss"])
            history["test_acc"].append(test["accuracy"])

            history["rate1"].append(train["rate1"])
            history["rate2"].append(train["rate2"])

            current_epoch = epoch + 1

            # --------------------------------------------------
            # Save best model
            # --------------------------------------------------

            if test["accuracy"] > best_test_accuracy:

                best_test_accuracy = test["accuracy"]
                best_epoch = current_epoch

                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }

                best_labels = test["labels"].clone()
                best_predictions = test["predictions"].clone()

                print(
                    f"\n*** New best model: "
                    f"epoch {best_epoch}, "
                    f"test accuracy = "
                    f"{100 * best_test_accuracy:.2f}% ***"
                )

            print()

            print("=" * 60)

            print(f"Epoch {current_epoch}/{epochs}")

            print(f"Train Loss : {train['loss']:.4f}")
            print(f"Train Acc  : {100 * train['accuracy']:.2f}%")

            print(f"Test Loss  : {test['loss']:.4f}")
            print(f"Test Acc   : {100 * test['accuracy']:.2f}%")

            print(f"Layer1 Rate: {100 * train['rate1']:.2f}%")
            print(f"Layer2 Rate: {100 * train['rate2']:.2f}%")

            print("=" * 60)

        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_test_accuracy": best_test_accuracy,
            "best_state_dict": best_state_dict,
            "best_labels": best_labels,
            "best_predictions": best_predictions,
        }

    #################################################################

    def save(self, path):

        torch.save(
            self.model.state_dict(),
            path,
        )

    #################################################################

    def load(self, path):

        self.model.load_state_dict(
            torch.load(
                path,
                map_location=self.device,
            )
        )