
"""
Readout classifiers for spiking neural networks.

The classifier receives spike activity
and converts it into class predictions.

Decoding strategy:

spike count / firing rate
        |
        v
neural network classifier
        |
        v
logits
"""

import torch
import torch.nn as nn


# ============================================================
# Spike Count Decoder
# ============================================================

class SpikeCountDecoder(nn.Module):
    """
    Converts spike trains into firing-rate features.

    Input:
        spikes:
            [time, batch, neurons]

    Output:
        features:
            [batch, neurons]
    """

    def __init__(
        self,
        normalize=True,
    ):
        super().__init__()

        self.normalize = normalize

    def forward(
        self,
        spikes,
    ):
        """
        Convert spike trains to spike-count / firing-rate features.
        """

        # Total number of spikes for each neuron
        # across the complete simulation time.
        spike_count = spikes.sum(
            dim=0
        )

        # Convert spike count to average firing rate.
        if self.normalize:

            time_steps = spikes.shape[0]

            spike_count = (
                spike_count
                / time_steps
            )

        return spike_count


# ============================================================
# SNN Classifier
# ============================================================

class SNNClassifier(nn.Module):
    """
    MLP classifier for SNN outputs.

    Example:

        100 neurons
             |
             v
        Linear(100, 64)
             |
            ReLU
             |
          Dropout
             |
        Linear(64, 20)
             |
             v
           logits
    """

    def __init__(
        self,
        input_size,
        n_classes,
        hidden_size=64,
        dropout=0.3,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Spike decoder
        # ----------------------------------------------------

        self.decoder = SpikeCountDecoder(
            normalize=True
        )

        # ----------------------------------------------------
        # MLP classifier
        # ----------------------------------------------------

        self.network = nn.Sequential(

            nn.Linear(
                input_size,
                hidden_size,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout,
            ),

            nn.Linear(
                hidden_size,
                n_classes,
            ),
        )

    def forward(
        self,
        spikes,
    ):
        """
        Parameters
        ----------
        spikes:
            Spike activity with shape
            [time, batch, neurons].

        Returns
        -------
        logits:
            Class scores with shape
            [batch, n_classes].
        """

        # ----------------------------------------------------
        # Decode spike activity
        # ----------------------------------------------------

        features = self.decoder(
            spikes
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        logits = self.network(
            features
        )

        return logits
