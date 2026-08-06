from torch.utils.data import DataLoader

from .shd import SHDDataset


def get_dataloaders(
    root="./data",
    batch_size=32,
    time_steps=200,
):

    train_dataset = SHDDataset(
        root=root,
        train=True,
        time_steps=time_steps,
    )

    test_dataset = SHDDataset(
        root=root,
        train=False,
        time_steps=time_steps,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, test_loader