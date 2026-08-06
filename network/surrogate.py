"""
Surrogate gradient functions for spiking neural networks.

Spikes are non-differentiable:
    spike = 1 if membrane >= threshold else 0

To train SNNs with gradient descent we replace
the derivative of the spike function with a smooth approximation.
"""

import torch


class SurrogateSpike(torch.autograd.Function):
    """
    Fast sigmoid / triangular surrogate gradient.

    Forward:
        returns binary spike

    Backward:
        approximates gradient around threshold
    """

    @staticmethod
    def forward(ctx, membrane, sigma=1.0):
        """
        membrane:
            v - threshold

        positive -> spike
        negative -> no spike
        """

        ctx.save_for_backward(membrane)
        ctx.sigma = sigma

        return (membrane >= 0).float()


    @staticmethod
    def backward(ctx, grad_output):

        membrane, = ctx.saved_tensors
        sigma = ctx.sigma

        # triangular surrogate derivative
        grad = (
            1 - torch.abs(membrane) / sigma
        ).clamp(min=0)

        return grad_output * grad / sigma, None



def spike_fn(x, sigma=1.0):
    """
    User-friendly wrapper.
    """

    return SurrogateSpike.apply(x, sigma)