"""
Spike-Timing-Dependent Plasticity (STDP)

Learning rules are separated from neural dynamics.

Supported:
- Pair STDP
- Triplet STDP
- Homeostatic modulation
- Weight clipping
"""

import torch
import torch.nn as nn


class STDP(nn.Module):
    """
    STDP learning rule.

    Weight update:

    pre before post:
        potentiation (+)

    post before pre:
        depression (-)

    """

    def __init__(
        self,
        a_plus=0.01,
        a_minus=0.012,
        tau_plus=20.0,
        tau_minus=20.0,
        w_min=0.0,
        w_max=1.0,

        # Triplet STDP
        triplet=False,
        a3_plus=None,
        a3_minus=None,
        tau_pre_slow=100.0,
        tau_post_slow=100.0,

    ):
        super().__init__()

        self.a_plus = a_plus
        self.a_minus = a_minus

        self.tau_plus = tau_plus
        self.tau_minus = tau_minus

        self.w_min = w_min
        self.w_max = w_max


        self.triplet = triplet

        self.a3_plus = (
            a3_plus
            if a3_plus is not None
            else a_plus * 0.5
        )

        self.a3_minus = (
            a3_minus
            if a3_minus is not None
            else a_minus * 0.5
        )

        self.tau_pre_slow = tau_pre_slow
        self.tau_post_slow = tau_post_slow



    def forward(
        self,
        weights,
        pre_spikes,
        post_spikes,
        homeostasis=None,
        dt=1.0
    ):
        """
        Calculate new weights.

        Args:

            weights:
                [input, neurons]

            pre_spikes:
                [time, input]

            post_spikes:
                [time, neurons]

        Returns:

            updated weights
        """

        device = weights.device


        n_pre = weights.shape[0]
        n_post = weights.shape[1]


        delta_w = torch.zeros(
            n_pre,
            n_post,
            device=device
        )


        pre_trace = torch.zeros(
            n_pre,
            device=device
        )

        post_trace = torch.zeros(
            n_post,
            device=device
        )


        pre_slow = torch.zeros(
            n_pre,
            device=device
        )

        post_slow = torch.zeros(
            n_post,
            device=device
        )


        decay_pre = torch.exp(
            torch.tensor(
                -dt / self.tau_plus,
                device=device
            )
        )

        decay_post = torch.exp(
            torch.tensor(
                -dt / self.tau_minus,
                device=device
            )
        )


        decay_pre_slow = torch.exp(
            torch.tensor(
                -dt / self.tau_pre_slow,
                device=device
            )
        )

        decay_post_slow = torch.exp(
            torch.tensor(
                -dt / self.tau_post_slow,
                device=device
            )
        )


        time_steps = pre_spikes.shape[0]


        for t in range(time_steps):

            pre_t = pre_spikes[t]
            post_t = post_spikes[t]


            # update traces

            pre_trace = (
                pre_trace * decay_pre
                +
                pre_t
            )

            post_trace = (
                post_trace * decay_post
                +
                post_t
            )


            if self.triplet:

                pre_slow = (
                    pre_slow * decay_pre_slow
                    +
                    pre_t
                )

                post_slow = (
                    post_slow * decay_post_slow
                    +
                    post_t
                )


            # potentiation

            active_pre = pre_t > 0


            if active_pre.any():

                potentiation = (
                    self.a_plus *
                    post_trace
                )


                if self.triplet:

                    potentiation += (
                        self.a3_plus *
                        post_slow
                    )


                for i in torch.where(active_pre)[0]:

                    delta_w[i] += potentiation



            # depression

            active_post = post_t > 0


            if active_post.any():

                depression = (
                    self.a_minus *
                    pre_trace
                )


                if self.triplet:

                    depression += (
                        self.a3_minus *
                        pre_slow
                    )


                for j in torch.where(active_post)[0]:

                    delta_w[:, j] -= depression



        # homeostatic scaling

        if homeostasis is not None:

            delta_w *= homeostasis


        print(
            "delta_w:",
            delta_w.min().item(),
            delta_w.mean().item(),
            delta_w.max().item()
        )

        total_spikes = (
            pre_spikes.sum()
            +
            post_spikes.sum()
        )

        if total_spikes > 0:
            delta_w = delta_w / time_steps
            delta_w = delta_w / total_spikes

        weights = weights + delta_w


        # biological limits

        weights = torch.clamp(
            weights,
            self.w_min,
            self.w_max
        )


        return weights