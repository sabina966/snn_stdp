"""
Readout classifiers for spiking neural networks.

The classifier receives spike activity
and converts it into class predictions.

Decoding strategy:
    spike count / firing rate
        ->
    neural network classifier
        ->
    logits
"""


import torch
import torch.nn as nn



class SpikeCountDecoder(nn.Module):
    """
    Converts spike trains into firing-rate features.

    Input:

        spikes:
            [time, batch, neurons]


    Output:

        features:
            [batch, neurons]
    """

    def __init__(
        self,
        normalize=True
    ):
        super().__init__()

        self.normalize = normalize



    def forward(
        self,
        spikes
    ):

        # total spikes per neuron

        spike_count = spikes.sum(
            dim=0
        )


        if self.normalize:

            time_steps = spikes.shape[0]

            spike_count = (
                spike_count /
                time_steps
            )


        return spike_count





class SNNClassifier(nn.Module):
    """
    MLP classifier for SNN outputs.


    Example:

        100 neurons

             |
             v

        Linear(100,64)

             |

           ReLU

             |

        Linear(64,20)

    """

    def __init__(
        self,
        input_size,
        n_classes,
        hidden_size=64,
        dropout=0.3
    ):
        super().__init__()


        self.decoder = SpikeCountDecoder()



        self.network = nn.Sequential(

            nn.Linear(
                input_size,
                hidden_size
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_size,
                n_classes
            )

        )




    def forward(
        self,
        spikes
    ):

        features = self.decoder(
            spikes
        )


        logits = self.network(
            features
        )


        return logits