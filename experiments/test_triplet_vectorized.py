"""
Check correctness of the vectorized Triplet STDP implementation.
"""

import torch

from network.triplet.triplet_stdp import TripletSTDP


def reference_triplet_stdp(
    weights,
    pre_spikes,
    post_spikes,
    a2_plus,
    a2_minus,
    a3_plus,
    a3_minus,
    tau_plus,
    tau_minus,
    tau_x,
    tau_y,
    w_min,
    w_max,
    dt=1.0,
):
    """
    Slow reference implementation.

    This reproduces the original Python-loop implementation.
    """

    weights = weights.clone()

    n_pre = weights.shape[0]
    n_post = weights.shape[1]

    device = weights.device
    dtype = weights.dtype

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

    decay_r1 = torch.exp(
        torch.tensor(
            -dt / tau_plus,
            device=device,
            dtype=dtype,
        )
    )

    decay_r2 = torch.exp(
        torch.tensor(
            -dt / tau_x,
            device=device,
            dtype=dtype,
        )
    )

    decay_o1 = torch.exp(
        torch.tensor(
            -dt / tau_minus,
            device=device,
            dtype=dtype,
        )
    )

    decay_o2 = torch.exp(
        torch.tensor(
            -dt / tau_y,
            device=device,
            dtype=dtype,
        )
    )

    delta_w = torch.zeros(
        n_pre,
        n_post,
        device=device,
        dtype=dtype,
    )

    for t in range(pre_spikes.shape[0]):

        pre_t = pre_spikes[t]
        post_t = post_spikes[t]

        active_pre = pre_t > 0
        active_post = post_t > 0

        if active_post.any():

            for j in torch.where(active_post)[0]:

                potentiation = (
                    a2_plus * r1
                    + a3_plus * r1 * o2[j]
                )

                delta_w[:, j] += potentiation

        if active_pre.any():

            for i in torch.where(active_pre)[0]:

                depression = (
                    a2_minus * o1
                    + a3_minus * o1 * r2[i]
                )

                delta_w[i, :] -= depression

        r1 = r1 * decay_r1 + pre_t
        r2 = r2 * decay_r2 + pre_t

        o1 = o1 * decay_o1 + post_t
        o2 = o2 * decay_o2 + post_t

    total_spikes = (
        pre_spikes.sum()
        + post_spikes.sum()
    )

    if total_spikes > 0:
        delta_w /= total_spikes

    return torch.clamp(
        weights + delta_w,
        w_min,
        w_max,
    )


def main():

    torch.manual_seed(42)

    n_pre = 20
    n_post = 10
    time_steps = 50

    weights = (
        torch.rand(n_pre, n_post)
        * 0.5
    )

    pre_spikes = (
        torch.rand(time_steps, n_pre)
        < 0.15
    ).float()

    post_spikes = (
        torch.rand(time_steps, n_post)
        < 0.20
    ).float()

    params = dict(
        a2_plus=0.001,
        a2_minus=0.0012,

        a3_plus=0.0005,
        a3_minus=0.0006,

        tau_plus=20.0,
        tau_minus=20.0,

        tau_x=100.0,
        tau_y=100.0,

        w_min=0.0,
        w_max=0.5,
    )

    print("=" * 60)
    print("VECTORIZED TRIPLET STDP TEST")
    print("=" * 60)

    # --------------------------------------------------
    # Reference
    # --------------------------------------------------

    reference_weights = reference_triplet_stdp(
        weights,
        pre_spikes,
        post_spikes,
        **params,
    )

    # --------------------------------------------------
    # Vectorized
    # --------------------------------------------------

    triplet = TripletSTDP(**params)

    vectorized_weights = triplet(
        weights.clone(),
        pre_spikes,
        post_spikes,
    )

    # --------------------------------------------------
    # Compare
    # --------------------------------------------------

    difference = (
        reference_weights
        - vectorized_weights
    ).abs()

    max_difference = (
        difference.max().item()
    )

    mean_difference = (
        difference.mean().item()
    )

    print()
    print(
        f"Maximum difference : "
        f"{max_difference:.12e}"
    )

    print(
        f"Mean difference    : "
        f"{mean_difference:.12e}"
    )

    # --------------------------------------------------
    # Assertion
    # --------------------------------------------------

    if torch.allclose(
        reference_weights,
        vectorized_weights,
        atol=1e-7,
        rtol=1e-6,
    ):

        print()
        print("✓ TEST PASSED")
        print(
            "Vectorized Triplet STDP "
            "matches the reference implementation."
        )

    else:

        print()
        print("✗ TEST FAILED")

        raise AssertionError(
            "Vectorized and reference "
            "Triplet STDP results differ."
        )


if __name__ == "__main__":
    main()