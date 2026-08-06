"""
Homeostatic regulation for spiking neural networks.

Maintains target firing rate by adjusting
learning strength.

Inspired by biological synaptic homeostasis.
"""

import torch
import torch.nn as nn



class Homeostasis(nn.Module):
    """
    Firing-rate homeostasis.

    If neuron fires too much:
        decrease learning factor

    If neuron fires too little:
        increase learning factor
    """


    def __init__(
        self,
        n_neurons,
        target_rate=10.0,
        tau_homeostasis=5000.0,
        strength=0.02,
        min_factor=0.5,
        max_factor=2.0,
        dt=1.0,
    ):
        super().__init__()


        self.n_neurons = n_neurons

        self.target_rate = target_rate

        self.tau_homeostasis = tau_homeostasis

        self.strength = strength

        self.min_factor = min_factor
        self.max_factor = max_factor

        self.dt = dt



        # running firing rate

        self.register_buffer(
            "running_rate",
            torch.zeros(n_neurons)
        )


        # synaptic modulation factor

        self.register_buffer(
            "factor",
            torch.ones(n_neurons)
        )



    @torch.no_grad()
    def update(
        self,
        spikes
    ):
        """
        Update firing statistics.

        spikes:

            [time, neurons]

        """

        time_steps = spikes.shape[0]


        # number of spikes per neuron

        spike_count = spikes.sum(
            dim=0
        )


        # convert to Hz

        firing_rate = (
            spike_count /
            (time_steps * self.dt / 1000.0)
        )



        # exponential moving average

        decay = torch.exp(
            torch.tensor(
                -self.dt /
                self.tau_homeostasis,
                device=spikes.device
            )
        )


        self.running_rate = (
            decay *
            self.running_rate
            +
            (1 - decay) *
            firing_rate
        )



        # difference from target

        error = (
            self.target_rate
            -
            self.running_rate
        )


        # update factor

        self.factor = (
            1.0
            +
            self.strength *
            error /
            (self.target_rate + 1e-8)
        )


        self.factor = torch.clamp(
            self.factor,
            self.min_factor,
            self.max_factor
        )



    def get_factor(self):
        """
        Return current modulation factor.
        """

        return self.factor