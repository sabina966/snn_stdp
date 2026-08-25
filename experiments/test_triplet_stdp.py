
"""
Unit tests for Triplet STDP.

This test verifies that when the triplet terms are enabled
with non-zero coefficients:

    A3+ > 0
    A3- > 0

Triplet STDP produces a different synaptic-weight update
than the existing Pair STDP implementation.
"""

import torch

from network.stdp import STDP
from network.triplet.triplet_stdp import TripletSTDP


def main():

    torch.manual_seed(42)

    # ==================================================
    # Test configuration
    # ==================================================

    n_pre = 5
    n_post = 3
    time_steps = 20

    print("=" * 60)
    print("TRIPLET STDP UNIT TEST")
    print("=" * 60)

    # ==================================================
    # Random weights
    # ==================================================

    weights = torch.rand(
        n_pre,
        n_post,
    ) * 0.5

    # ==================================================
    # Synthetic spike trains
    # ==================================================

    pre_spikes = (
        torch.rand(
            time_steps,
            n_pre,
        ) < 0.15
    ).float()

    post_spikes = (
        torch.rand(
            time_steps,
            n_post,
        ) < 0.20
    ).float()

    print(
        "\nPre spikes :",
        int(pre_spikes.sum().item()),
    )

    print(
        "Post spikes:",
        int(post_spikes.sum().item()),
    )

    # ==================================================
    # Parameters
    # ==================================================

    a_plus = 0.001
    a_minus = 0.0012

    tau_plus = 20.0
    tau_minus = 20.0

    w_min = 0.0
    w_max = 0.5

    # ==================================================
    # Pair STDP
    # ==================================================

    pair_stdp = STDP(
        a_plus=a_plus,
        a_minus=a_minus,
        tau_plus=tau_plus,
        tau_minus=tau_minus,
        w_min=w_min,
        w_max=w_max,
    )

    pair_weights = pair_stdp(
        weights.clone(),
        pre_spikes,
        post_spikes,
    )

    # ==================================================
    # Triplet STDP
    #
    # Triplet terms ENABLED:
    #
    # A3+ = 0.001
    # A3- = 0.0012
    #
    # Therefore Triplet STDP should produce a different
    # result from Pair STDP.
    # ==================================================

    triplet_stdp = TripletSTDP(
        a2_plus=a_plus,
        a2_minus=a_minus,

        a3_plus=0.001,
        a3_minus=0.0012,

        tau_plus=tau_plus,
        tau_minus=tau_minus,

        tau_x=100.0,
        tau_y=100.0,

        w_min=w_min,
        w_max=w_max,
    )

    triplet_weights = triplet_stdp(
        weights.clone(),
        pre_spikes,
        post_spikes,
    )

    # ==================================================
    # Compare
    # ==================================================

    difference = (
        pair_weights - triplet_weights
    ).abs()

    max_difference = difference.max().item()
    mean_difference = difference.mean().item()

    print()
    print("=" * 60)
    print("TRIPLET EFFECT")
    print("=" * 60)

    print(
        f"Maximum difference : "
        f"{max_difference:.12e}"
    )

    print(
        f"Mean difference    : "
        f"{mean_difference:.12e}"
    )

    # ==================================================
    # Test
    # ==================================================

    if torch.allclose(
        pair_weights,
        triplet_weights,
        atol=1e-7,
        rtol=1e-6,
    ):

        print()
        print("✗ TEST FAILED")

        print(
            "Triplet coefficients are non-zero, "
            "but Pair and Triplet STDP are identical."
        )

        raise AssertionError(
            "Triplet terms do not affect the weights."
        )

    else:

        print()
        print("✓ TEST PASSED")

        print(
            "Non-zero triplet coefficients "
            "change the synaptic weights."
        )


if __name__ == "__main__":
    main()
