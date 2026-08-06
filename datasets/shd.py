"""
SHD dataset utilities.
"""

from torch.utils.data import Dataset
import torch
import tonic


class SHDDataset(Dataset):
    """
    Wrapper around Tonic SHD dataset.

    Returns
    -------
    spikes : Tensor
        Shape [time, input]
    label : int
    """

    def __init__(
        self,
        root="./data",
        train=True,
        time_steps=200,
        n_input=700,
    ):

        self.time_steps = time_steps
        self.n_input = n_input

        self.dataset = tonic.datasets.SHD(
            save_to=root,
            train=train,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        events, label = self.dataset[index]

        spikes = self.events_to_spikes(events)

        return spikes.float(), torch.tensor(label, dtype=torch.long)

    def events_to_spikes(self, events):

        spikes = torch.zeros(
            self.time_steps,
            self.n_input,
            dtype=torch.float32,
        )

        t = torch.as_tensor(events["t"])

        x = torch.as_tensor(events["x"]).long()

        t0 = t.min()
        t1 = t.max()

        if t1 == t0:
            return spikes

        t = (
            (t - t0).float()
            /
            (t1 - t0).float()
        )

        t = (
            t * (self.time_steps - 1)
        ).long()

        spikes[t, x] = 1.0

        return spikes