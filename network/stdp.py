"""
Spike-Timing-Dependent Plasticity (STDP)

Pure Pair STDP.

pre before post  -> potentiation (LTP)
post before pre  -> depression (LTD)

Weights are clipped to biological bounds.
"""

import torch
import torch.nn as nn


class STDP(nn.Module):

    def __init__(
        self,
        a_plus=0.001,
        a_minus=0.0012,
        tau_plus=20.0,
        tau_minus=20.0,
        w_min=0.0,
        w_max=0.5,
    ):
        super().__init__()

        self.a_plus = a_plus
        self.a_minus = a_minus

        self.tau_plus = tau_plus
        self.tau_minus = tau_minus

        self.w_min = w_min
        self.w_max = w_max

    def forward(
        self,
        weights,
        pre_spikes,
        post_spikes,
        homeostasis=None,
        dt=1.0,
    ):
        """
        Apply pair STDP.

        weights:
            [input, neurons]

        pre_spikes:
            [time, input]

        post_spikes:
            [time, neurons]
        """

        device = weights.device

        n_pre = weights.shape[0]
        n_post = weights.shape[1]

        time_steps = pre_spikes.shape[0]

        # --------------------------------------------------
        # Traces
        # --------------------------------------------------

        pre_trace = torch.zeros(
            n_pre,
            device=device,
        )

        post_trace = torch.zeros(
            n_post,
            device=device,
        )

        # --------------------------------------------------
        # Trace decay
        # --------------------------------------------------

        decay_pre = torch.exp(
            torch.tensor(
                -dt / self.tau_plus,
                device=device,
            )
        )

        decay_post = torch.exp(
            torch.tensor(
                -dt / self.tau_minus,
                device=device,
            )
        )

        # --------------------------------------------------
        # Weight update
        # --------------------------------------------------

        delta_w = torch.zeros(
            n_pre,
            n_post,
            device=device,
        )

        # ==================================================
        # Time loop
        # ==================================================

        for t in range(time_steps):

            pre_t = pre_spikes[t]
            post_t = post_spikes[t]

            # --------------------------------------------------
            # LTP
            #
            # pre spike happens after previous pre activity,
            # and current post spike sees the pre trace.
            # --------------------------------------------------

            active_post = post_t > 0

            if active_post.any():

                potentiation = (
                    self.a_plus * pre_trace
                )

                for j in torch.where(active_post)[0]:

                    delta_w[:, j] += potentiation

            # --------------------------------------------------
            # LTD
            #
            # current pre spike sees previous post activity.
            # --------------------------------------------------

            active_pre = pre_t > 0

            if active_pre.any():

                depression = (
                    self.a_minus * post_trace
                )

                for i in torch.where(active_pre)[0]:

                    delta_w[i, :] -= depression

            # --------------------------------------------------
            # Update traces
            # --------------------------------------------------

            pre_trace = (
                pre_trace * decay_pre
                + pre_t
            )

            post_trace = (
                post_trace * decay_post
                + post_t
            )

        # ==================================================
        # Homeostasis
        # ==================================================

        if homeostasis is not None:

            delta_w *= homeostasis

        # --------------------------------------------------
        # Normalize update
        # --------------------------------------------------

        total_spikes = (
            pre_spikes.sum()
            + post_spikes.sum()
        )

        if total_spikes > 0:

            delta_w /= total_spikes

        # --------------------------------------------------
        # Debug
        # --------------------------------------------------

        print(
            "delta_w:",
            delta_w.min().item(),
            delta_w.mean().item(),
            delta_w.max().item(),
        )

        # ==================================================
        # Update weights
        # ==================================================

        weights = weights + delta_w

        # ==================================================
        # Biological bounds
        # ==================================================

        weights = torch.clamp(
            weights,
            self.w_min,
            self.w_max,
        )

        return weights