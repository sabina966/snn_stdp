"""
Neuron models for Spiking Neural Networks.
"""

import torch
import torch.nn as nn

from .surrogate import spike_fn


class LIFNeuron(nn.Module):
    """
    Adaptive Leaky Integrate-and-Fire neuron.

    This class performs ONE simulation step.
    The simulation loop is handled by SpikingLinear.
    """

    def __init__(
        self,
        tau_m=20.0,
        v_rest=-65.0,
        v_reset=-65.0,
        v_threshold=-50.0,
        tau_adaptation=100.0,
        adaptation_strength=0.5,
        dt=1.0,
        surrogate_sigma=1.0,
    ):
        super().__init__()

        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_threshold = v_threshold

        self.tau_adaptation = tau_adaptation
        self.adaptation_strength = adaptation_strength

        self.dt = dt
        self.surrogate_sigma = surrogate_sigma

    def initialize_state(self, batch_size, n_neurons, device):

        voltage = torch.full(
            (batch_size, n_neurons),
            self.v_rest,
            device=device,
        )

        adaptation = torch.zeros(
            batch_size,
            n_neurons,
            device=device,
        )

        return voltage, adaptation

    def forward_step(
        self,
        input_current,
        voltage,
        adaptation,
    ):
        """
        One simulation step.

        Parameters
        ----------
        input_current
            [batch, neurons]

        voltage
            [batch, neurons]

        adaptation
            [batch, neurons]
        """

        threshold = self.v_threshold + adaptation

        dv = (
            -(voltage - self.v_rest)
            + input_current
        ) * (self.dt / self.tau_m)

        voltage = voltage + dv

        spike = spike_fn(
            voltage - threshold,
            self.surrogate_sigma,
        )

        voltage = torch.where(
            spike > 0,
            torch.full_like(voltage, self.v_reset),
            voltage,
        )

        adaptation = (
            adaptation
            * (1 - self.dt / self.tau_adaptation)
            + spike * self.adaptation_strength
        )

        return spike, voltage, adaptation