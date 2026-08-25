
"""
Frequency-dependence test for Triplet STDP.

The test compares two spike patterns:

Pattern A:
    pre -> post

Pattern B:
    pre -> post -> post

The second postsynaptic spike occurs while the slow
postsynaptic trace is still active.

Therefore, with non-zero A3+:

    A3+ * r1 * o2

should produce an additional potentiation in Pattern B.

This demonstrates the frequency-dependent component
of Triplet STDP.
"""

import torch

from network.stdp import STDP
from network.triplet.triplet_stdp import TripletSTDP


def run_pair(
    stdp,
    weights,
    pre_spikes,
    post_spikes,
):
    return stdp(
        weights.clone(),
        pre_spikes,
        post_spikes,
    )


def run_triplet(
    stdp,
    weights,
    pre_spikes,
    post_spikes,
):
    return stdp(
        weights.clone(),
        pre_spikes,
        post_spikes,
    )


def main():

    torch.manual_seed(42)

    print("=" * 60)
    print("TRIPLET STDP FREQUENCY TEST")
    print("=" * 60)

    # ==================================================
    # Configuration
    # ==================================================

    n_pre = 1
    n_post = 1
    time_steps = 20

    dt = 1.0

    initial_weight = 0.25

    a2_plus = 0.001
    a2_minus = 0.0012

    a3_plus = 0.001
    a3_minus = 0.0012

    tau_plus = 20.0
    tau_minus = 20.0

    tau_x = 100.0
    tau_y = 100.0

    w_min = 0.0
    w_max = 0.5

    # ==================================================
    # Initial weight
    # ==================================================

    weights = torch.tensor(
        [[initial_weight]],
        dtype=torch.float32,
    )

    # ==================================================
    # Spike pattern A
    #
    # pre -> post
    # ==================================================

    pre_a = torch.zeros(
        time_steps,
        n_pre,
    )

    post_a = torch.zeros(
        time_steps,
        n_post,
    )

    pre_a[5, 0] = 1.0
    post_a[6, 0] = 1.0

    # ==================================================
    # Spike pattern B
    #
    # pre -> post -> post
    #
    # The second post spike occurs while o2 is active.
    # ==================================================

    pre_b = torch.zeros(
        time_steps,
        n_pre,
    )

    post_b = torch.zeros(
        time_steps,
        n_post,
    )

    pre_b[5, 0] = 1.0
    post_b[6, 0] = 1.0
    post_b[10, 0] = 1.0

    # ==================================================
    # Pair STDP
    # ==================================================

    pair_stdp = STDP(
        a_plus=a2_plus,
        a_minus=a2_minus,
        tau_plus=tau_plus,
        tau_minus=tau_minus,
        w_min=w_min,
        w_max=w_max,
    )

    pair_a = run_pair(
        pair_stdp,
        weights,
        pre_a,
        post_a,
    )

    pair_b = run_pair(
        pair_stdp,
        weights,
        pre_b,
        post_b,
    )

    # ==================================================
    # Triplet STDP
    # ==================================================

    triplet_stdp = TripletSTDP(
        a2_plus=a2_plus,
        a2_minus=a2_minus,

        a3_plus=a3_plus,
        a3_minus=a3_minus,

        tau_plus=tau_plus,
        tau_minus=tau_minus,

        tau_x=tau_x,
        tau_y=tau_y,

        w_min=w_min,
        w_max=w_max,
    )

    triplet_a = run_triplet(
        triplet_stdp,
        weights,
        pre_a,
        post_a,
    )

    triplet_b = run_triplet(
        triplet_stdp,
        weights,
        pre_b,
        post_b,
    )

    # ==================================================
    # Weight changes
    # ==================================================

    pair_delta_a = (
        pair_a - weights
    ).item()

    pair_delta_b = (
        pair_b - weights
    ).item()

    triplet_delta_a = (
        triplet_a - weights
    ).item()

    triplet_delta_b = (
        triplet_b - weights
    ).item()

    # ==================================================
    # Additional effect of the second post spike
    # ==================================================

    pair_extra = (
        pair_delta_b - pair_delta_a
    )

    triplet_extra = (
        triplet_delta_b - triplet_delta_a
    )

    # ==================================================
    # Print results
    # ==================================================

    print()
    print("Pattern A: pre -> post")
    print(
        f"Pair delta     : {pair_delta_a:.12e}"
    )
    print(
        f"Triplet delta  : {triplet_delta_a:.12e}"
    )

    print()
    print("Pattern B: pre -> post -> post")
    print(
        f"Pair delta     : {pair_delta_b:.12e}"
    )
    print(
        f"Triplet delta  : {triplet_delta_b:.12e}"
    )

    print()
    print("=" * 60)
    print("EFFECT OF ADDITIONAL POSTSYNAPTIC SPIKE")
    print("=" * 60)

    print(
        f"Pair extra effect    : "
        f"{pair_extra:.12e}"
    )

    print(
        f"Triplet extra effect : "
        f"{triplet_extra:.12e}"
    )

    # ==================================================
    # Test
    # ==================================================

    if triplet_extra <= pair_extra:

        print()
        print("✗ TEST FAILED")

        print(
            "The additional postsynaptic spike does not "
            "produce the expected extra triplet effect."
        )

        raise AssertionError(
            "Triplet frequency-dependent effect "
            "was not detected."
        )

    print()
    print("✓ TEST PASSED")

    print(
        "Triplet STDP shows an additional "
        "frequency-dependent effect."
    )


if __name__ == "__main__":
    main()

