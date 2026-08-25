"""
Triplet-STDP spiking layer.

This module is intentionally separated from the original
Pair-STDP SpikingLinear implementation.

The original network/layers.py is NOT modified.
"""

import torch
import torch.nn as nn

from ..neuron import LIFNeuron
from ..homeostasis import Homeostasis
from .triplet_stdp import TripletSTDP


class TripletSpikingLinear(nn.Module):
    """
    Linear spiking layer with Triplet STDP.

    Input:
        [time, batch, input]

    Output:
        [time, batch, neurons]
    """

    def __init__(
        self,
        n_input,
        n_neurons,
        neuron_params=None,
        triplet_params=None,
        homeostasis_params=None,
        input_gain=100.0,
        use_bias=False,
    ):
        super().__init__()

        if neuron_params is None:
            neuron_params = {}

        if triplet_params is None:
            triplet_params = {}

        if homeostasis_params is None:
            homeostasis_params = {}

        self.n_input = n_input
        self.n_neurons = n_neurons

        self.input_gain = input_gain

        # ==================================================
        # Synaptic weights
        # ==================================================

        self.weights = nn.Parameter(
            torch.empty(
                n_input,
                n_neurons,
            )
        )

        nn.init.uniform_(
            self.weights,
            0.0,
            0.5,
        )

        # ==================================================
        # Optional bias
        # ==================================================

        if use_bias:

            self.bias = nn.Parameter(
                torch.zeros(
                    n_neurons
                )
            )

        else:

            self.register_parameter(
                "bias",
                None,
            )

        # ==================================================
        # Neuron
        # ==================================================

        self.neuron = LIFNeuron(
            **neuron_params
        )

        # ==================================================
        # Triplet STDP
        # ==================================================

        self.stdp = TripletSTDP(
            **triplet_params
        )

        # ==================================================
        # Homeostasis
        # ==================================================

        self.homeostasis = Homeostasis(
            n_neurons=n_neurons,
            **homeostasis_params,
        )

        # ==================================================
        # Diagnostics
        # ==================================================

        self.last_ltp = None
        self.last_ltd = None
        self.last_delta_w = None

    def compute_current(
        self,
        input_spikes,
    ):
        """
        Calculate synaptic current.

        Parameters
        ----------
        input_spikes:
            [batch, input]

        Returns
        -------
        current:
            [batch, neurons]
        """

        current = (
            input_spikes
            @ self.weights
        )

        if self.bias is not None:

            current = (
                current
                + self.bias
            )

        return (
            current
            * self.input_gain
        )

    def forward(
        self,
        input_spikes,
        apply_stdp=False,
        return_voltage=False,
    ):
        """
        Parameters
        ----------
        input_spikes:
            [time, batch, input]

        apply_stdp:
            Whether to apply Triplet STDP.

        return_voltage:
            Whether to return membrane potentials.

        Returns
        -------
        spikes:
            [time, batch, neurons]

        or

        spikes, voltages
        """

        T, B, _ = input_spikes.shape

        # ==================================================
        # Initialize neuron state
        # ==================================================

        voltage, adaptation = (
            self.neuron.initialize_state(
                batch_size=B,
                n_neurons=self.n_neurons,
                device=input_spikes.device,
            )
        )

        spike_history = []

        voltage_history = []

        # ==================================================
        # Simulation
        # ==================================================

        for t in range(T):

            current = self.compute_current(
                input_spikes[t]
            )

            (
                spikes,
                voltage,
                adaptation,
            ) = self.neuron.forward_step(
                current,
                voltage,
                adaptation,
            )

            spike_history.append(
                spikes
            )

            voltage_history.append(
                voltage
            )

        spikes = torch.stack(
            spike_history
        )

        voltages = torch.stack(
            voltage_history
        )

        # ==================================================
        # Triplet STDP
        # ==================================================

        if apply_stdp:

            with torch.no_grad():

                # ------------------------------------------
                # Average over batch
                # ------------------------------------------

                pre = (
                    input_spikes
                    .mean(dim=1)
                )

                post = (
                    spikes
                    .mean(dim=1)
                )

                # ------------------------------------------
                # Homeostatic factor
                # ------------------------------------------

                factor = (
                    self.homeostasis.get_factor()
                )

                # ------------------------------------------
                # Update weights
                # ------------------------------------------

                updated_weights = self.stdp(
                    self.weights.data,
                    pre,
                    post,
                    factor,
                )

                self.weights.data.copy_(
                    updated_weights
                )

                # ------------------------------------------
                # Save diagnostics
                # ------------------------------------------

                self.last_ltp = (
                    self.stdp.last_ltp
                    .detach()
                    .clone()
                )

                self.last_ltd = (
                    self.stdp.last_ltd
                    .detach()
                    .clone()
                )

                self.last_delta_w = (
                    self.stdp.last_delta_w
                    .detach()
                    .clone()
                )

                # ------------------------------------------
                # Update homeostasis
                # ------------------------------------------

                self.homeostasis.update(
                    post
                )

        # ==================================================
        # Return
        # ==================================================

        if return_voltage:

            return (
                spikes,
                voltages,
            )

        return spikes