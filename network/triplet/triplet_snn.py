
"""
Hierarchical SNN with Triplet STDP.

This is a completely separate network from the original
Pair-STDP HierarchicalSNN.

The original network/hierarchical_snn.py is NOT modified.
"""

import torch
import torch.nn as nn

from .triplet_layers import TripletSpikingLinear
from ..classifier import SNNClassifier


class TripletHierarchicalSNN(nn.Module):
    """
    Two-layer hierarchical SNN using Triplet STDP.
    """

    def __init__(
        self,
        n_input=700,
        n_hidden1=200,
        n_hidden2=100,
        n_classes=20,
        neuron_params=None,
        triplet_params1=None,
        triplet_params2=None,
        homeostasis_params=None,
        input_gain=15.0,
        use_classifier=False,
    ):
        super().__init__()

        self.use_classifier = use_classifier

        # --------------------------------------------------
        # Layer 1
        # --------------------------------------------------

        self.layer1 = TripletSpikingLinear(
            n_input=n_input,
            n_neurons=n_hidden1,
            neuron_params=neuron_params,
            triplet_params=triplet_params1,
            homeostasis_params=homeostasis_params,
            input_gain=input_gain,
        )

        # --------------------------------------------------
        # Layer 2
        # --------------------------------------------------

        self.layer2 = TripletSpikingLinear(
            n_input=n_hidden1,
            n_neurons=n_hidden2,
            neuron_params=neuron_params,
            triplet_params=triplet_params2,
            homeostasis_params=homeostasis_params,
            input_gain=input_gain,
        )

        # --------------------------------------------------
        # Optional classifier
        # --------------------------------------------------

        if self.use_classifier:

            self.classifier = SNNClassifier(
                input_size=n_hidden2,
                n_classes=n_classes,
            )

        else:

            self.classifier = None

    def forward(
        self,
        spikes,
        apply_stdp=False,
        return_activity=False,
    ):
        """
        Parameters
        ----------
        spikes:
            [time, batch, input]

        apply_stdp:
            Whether to apply Triplet STDP.

        return_activity:
            Whether to return Layer 1 and Layer 2 spikes.

        Returns
        -------
        With classifier:

            logits

        Without classifier:

            spikes2

        With return_activity=True:

            logits, spikes1, spikes2

        or

            spikes1, spikes2
        """

        # --------------------------------------------------
        # Layer 1
        # --------------------------------------------------

        spikes1 = self.layer1(
            spikes,
            apply_stdp=apply_stdp,
        )

        # --------------------------------------------------
        # Layer 2
        # --------------------------------------------------

        spikes2 = self.layer2(
            spikes1,
            apply_stdp=apply_stdp,
        )

        # --------------------------------------------------
        # Classifier
        # --------------------------------------------------

        if self.classifier is not None:

            logits = self.classifier(
                spikes2
            )

            if return_activity:

                return (
                    logits,
                    spikes1,
                    spikes2,
                )

            return logits

        # --------------------------------------------------
        # Pure Triplet STDP
        # --------------------------------------------------

        if return_activity:

            return (
                spikes1,
                spikes2,
            )

        return spikes2
