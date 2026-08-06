"""
Spiking layer implementation.

Combines:
- Linear synapses
- LIF neurons
- STDP (optional)
- Homeostasis (optional)
"""

import torch
import torch.nn as nn

from .neuron import LIFNeuron
from .stdp import STDP
from .homeostasis import Homeostasis


class SpikingLinear(nn.Module):

    def __init__(
        self,
        n_input,
        n_neurons,
        neuron_params=None,
        stdp_params=None,
        homeostasis_params=None,
        input_gain=100.0,
        use_bias=False,
    ):
        super().__init__()

        if neuron_params is None:
            neuron_params = {}

        if stdp_params is None:
            stdp_params = {}

        if homeostasis_params is None:
            homeostasis_params = {}

        self.n_input = n_input
        self.n_neurons = n_neurons

        self.input_gain = input_gain

        # ------------------------
        # Synaptic weights
        # ------------------------

        self.weights = nn.Parameter(
            torch.empty(n_input, n_neurons)
        )

        nn.init.uniform_(
            self.weights,
            0,
            0.5
        )

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(n_neurons))
        else:
            self.register_parameter("bias", None)

        # ------------------------
        # Components
        # ------------------------

        self.neuron = LIFNeuron(**neuron_params)

        self.stdp = STDP(**stdp_params)

        self.homeostasis = Homeostasis(
            n_neurons=n_neurons,
            **homeostasis_params
        )

    def compute_current(self, input_spikes):
        """
        input_spikes:
            [batch,input]

        returns:
            [batch,neurons]
        """

        current = input_spikes @ self.weights

        if self.bias is not None:
            current = current + self.bias

        return current * self.input_gain

    def forward(
        self,
        input_spikes,
        apply_stdp=False,
        return_voltage=False,
    ):
        """
        input_spikes:

            [time,batch,input]
        """

        T, B, _ = input_spikes.shape

        voltage, adaptation = self.neuron.initialize_state(
            batch_size=B,
            n_neurons=self.n_neurons,
            device=input_spikes.device,
        )

        spike_history = []
        voltage_history = []

        for t in range(T):

            current = self.compute_current(
                input_spikes[t]
            )

            spikes, voltage, adaptation = (
                self.neuron.forward_step(
                    current,
                    voltage,
                    adaptation,
                )
            )

            spike_history.append(spikes)
            voltage_history.append(voltage)

        spikes = torch.stack(spike_history)
        voltages = torch.stack(voltage_history)

        # ------------------------
        # Homeostasis
        # ------------------------

        if apply_stdp:

            with torch.no_grad():

                # average over batch
                pre = input_spikes.mean(dim=1)
                post = spikes.mean(dim=1)

                factor = self.homeostasis.get_factor()

                self.weights.data.copy_(
                    self.stdp(
                        self.weights.data,
                        pre,
                        post,
                        factor,
                    )
                )

                self.homeostasis.update(post)

        if return_voltage:
            return spikes, voltages

        return spikes