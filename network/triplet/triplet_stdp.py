"""
Triplet Spike-Timing-Dependent Plasticity (STDP).

Based on:

Pfister, J.-P. & Gerstner, W. (2006).
Triplets of Spikes in a Model of Spike-Timing-Dependent Plasticity.

Full All-to-All triplet rule.

Traces:
r1 -- fast presynaptic trace
r2 -- slow presynaptic trace
o1 -- fast postsynaptic trace
o2 -- slow postsynaptic trace

At a postsynaptic spike:

    dW += A2+ * r1
    dW += A3+ * r1 * o2

At a presynaptic spike:

    dW -= A2- * o1
    dW -= A3- * o1 * r2

The implementation is vectorized over pre/post neurons.

Additionally, the implementation stores diagnostics:

    last_ltp
    last_ltd
    last_delta_w

These values are used only for analysis and do NOT change
the STDP update itself.
"""

import torch
import torch.nn as nn


class TripletSTDP(nn.Module):

    def __init__(
        self,

        # --------------------------------------------------
        # Pair terms
        # --------------------------------------------------

        a2_plus=0.001,
        a2_minus=0.0012,

        # --------------------------------------------------
        # Triplet terms
        # --------------------------------------------------

        a3_plus=0.001,
        a3_minus=0.001,

        # --------------------------------------------------
        # Fast traces
        # --------------------------------------------------

        tau_plus=20.0,
        tau_minus=20.0,

        # --------------------------------------------------
        # Slow traces
        # --------------------------------------------------

        tau_x=100.0,
        tau_y=100.0,

        # --------------------------------------------------
        # Weight bounds
        # --------------------------------------------------

        w_min=0.0,
        w_max=0.5,
    ):
        super().__init__()

        self.a2_plus = a2_plus
        self.a2_minus = a2_minus

        self.a3_plus = a3_plus
        self.a3_minus = a3_minus

        self.tau_plus = tau_plus
        self.tau_minus = tau_minus

        self.tau_x = tau_x
        self.tau_y = tau_y

        self.w_min = w_min
        self.w_max = w_max

        # --------------------------------------------------
        # Diagnostics
        # --------------------------------------------------

        self.last_ltp = None
        self.last_ltd = None
        self.last_delta_w = None

    def forward(
        self,
        weights,
        pre_spikes,
        post_spikes,
        homeostasis=None,
        dt=1.0,
    ):
        """
        Apply full All-to-All Triplet STDP.

        Parameters
        ----------
        weights:
            [n_pre, n_post]

        pre_spikes:
            [time, n_pre]

        post_spikes:
            [time, n_post]

        homeostasis:
            Optional multiplicative homeostatic factor.

        dt:
            Simulation time step in ms.

        Returns
        -------
        updated_weights:
            [n_pre, n_post]

        Diagnostics are stored in:

            self.last_ltp
            self.last_ltd
            self.last_delta_w
        """

        device = weights.device
        dtype = weights.dtype

        n_pre = weights.shape[0]
        n_post = weights.shape[1]

        time_steps = pre_spikes.shape[0]

        # ==================================================
        # Traces
        # ==================================================

        r1 = torch.zeros(
            n_pre,
            device=device,
            dtype=dtype,
        )

        r2 = torch.zeros(
            n_pre,
            device=device,
            dtype=dtype,
        )

        o1 = torch.zeros(
            n_post,
            device=device,
            dtype=dtype,
        )

        o2 = torch.zeros(
            n_post,
            device=device,
            dtype=dtype,
        )

        # ==================================================
        # Trace decay
        # ==================================================

        decay_r1 = torch.exp(
            torch.tensor(
                -dt / self.tau_plus,
                device=device,
                dtype=dtype,
            )
        )

        decay_r2 = torch.exp(
            torch.tensor(
                -dt / self.tau_x,
                device=device,
                dtype=dtype,
            )
        )

        decay_o1 = torch.exp(
            torch.tensor(
                -dt / self.tau_minus,
                device=device,
                dtype=dtype,
            )
        )

        decay_o2 = torch.exp(
            torch.tensor(
                -dt / self.tau_y,
                device=device,
                dtype=dtype,
            )
        )

        # ==================================================
        # Separate LTP and LTD
        # ==================================================

        ltp = torch.zeros(
            n_pre,
            n_post,
            device=device,
            dtype=dtype,
        )

        ltd = torch.zeros(
            n_pre,
            n_post,
            device=device,
            dtype=dtype,
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
            # At postsynaptic spikes:
            #
            # A2+ * r1
            # A3+ * r1 * o2
            # --------------------------------------------------

            potentiation = (
                self.a2_plus * r1[:, None]
                +
                self.a3_plus
                * r1[:, None]
                * o2[None, :]
            )

            ltp += (
                potentiation
                * post_t[None, :]
            )

            # --------------------------------------------------
            # LTD
            #
            # At presynaptic spikes:
            #
            # A2- * o1
            # A3- * o1 * r2
            # --------------------------------------------------

            depression = (
                self.a2_minus * o1[None, :]
                +
                self.a3_minus
                * r2[:, None]
                * o1[None, :]
            )

            ltd += (
                depression
                * pre_t[:, None]
            )

            # --------------------------------------------------
            # Update traces
            #
            # Current spikes are added AFTER the weight update.
            # --------------------------------------------------

            r1 = (
                r1 * decay_r1
                + pre_t
            )

            r2 = (
                r2 * decay_r2
                + pre_t
            )

            o1 = (
                o1 * decay_o1
                + post_t
            )

            o2 = (
                o2 * decay_o2
                + post_t
            )

        # ==================================================
        # Net STDP update
        # ==================================================

        delta_w = ltp - ltd

        # ==================================================
        # Homeostasis
        # ==================================================

        if homeostasis is not None:

            ltp = ltp * homeostasis
            ltd = ltd * homeostasis

            delta_w = ltp - ltd

        # ==================================================
        # Normalize update
        # ==================================================

        total_spikes = (
            pre_spikes.sum()
            +
            post_spikes.sum()
        )

        if total_spikes > 0:

            ltp = ltp / total_spikes
            ltd = ltd / total_spikes
            delta_w = delta_w / total_spikes

        # ==================================================
        # Save diagnostics
        #
        # These are detached so they don't retain
        # computation graphs.
        # ==================================================

        self.last_ltp = ltp.detach().clone()

        self.last_ltd = ltd.detach().clone()

        self.last_delta_w = delta_w.detach().clone()

        # ==================================================
        # Update weights
        # ==================================================

        updated_weights = (
            weights
            + delta_w
        )

        # ==================================================
        # Biological bounds
        # ==================================================

        updated_weights = torch.clamp(
            updated_weights,
            self.w_min,
            self.w_max,
        )

        return updated_weights