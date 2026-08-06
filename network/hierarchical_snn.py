"""
Hierarchical Spiking Neural Network.

Architecture:

Input spikes
      |
      v
SpikingLinear layer 1
      |
      v
SpikingLinear layer 2
      |
      +------------------------------+
      |                              |
      | use_classifier=False         | use_classifier=True
      |                              |
      v                              v
Output spikes                  Spike decoder
                                     |
                                     v
                                Classifier


Supports:
- Pair STDP
- Triplet STDP
- STDP pretraining
- Surrogate gradient training
"""

import torch
import torch.nn as nn

from .layers import SpikingLinear
from .classifier import SNNClassifier


class HierarchicalSNN(nn.Module):
    """
    Two-layer hierarchical SNN.
    """

    def __init__(
        self,

        n_input=700,

        n_hidden1=200,

        n_hidden2=100,

        n_classes=20,

        neuron_params=None,

        stdp_params1=None,

        stdp_params2=None,

        homeostasis_params=None,

        input_gain=15.0,

        use_classifier=True,

    ):

        super().__init__()

        self.use_classifier = use_classifier

        # -------------------------------------------------
        # Layer 1
        # -------------------------------------------------

        self.layer1 = SpikingLinear(

            n_input=n_input,

            n_neurons=n_hidden1,

            neuron_params=neuron_params,

            stdp_params=stdp_params1,

            homeostasis_params=homeostasis_params,

            input_gain=input_gain,

        )

        # -------------------------------------------------
        # Layer 2
        # -------------------------------------------------

        self.layer2 = SpikingLinear(

            n_input=n_hidden1,

            n_neurons=n_hidden2,

            neuron_params=neuron_params,

            stdp_params=stdp_params2,

            homeostasis_params=homeostasis_params,

            input_gain=input_gain,

        )

        # -------------------------------------------------
        # Optional classifier
        # -------------------------------------------------

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
        spikes :
            Tensor [time, batch, input]

        Returns
        -------
        Hybrid mode:
            logits

        Pure STDP mode:
            spikes from second layer
        """

        # -------------------------------------------------
        # Layer 1
        # -------------------------------------------------

        spikes1 = self.layer1(

            spikes,

            apply_stdp=apply_stdp,

        )

        # -------------------------------------------------
        # Layer 2
        # -------------------------------------------------

        spikes2 = self.layer2(

            spikes1,

            apply_stdp=apply_stdp,

        )

        # -------------------------------------------------
        # Hybrid mode
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Pure STDP mode
        # -------------------------------------------------

        if return_activity:

            return (

                spikes1,

                spikes2,

            )

        return spikes2