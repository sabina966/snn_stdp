import torch

from configs import Config
from datasets import get_dataloaders
from network.hierarchical_snn import HierarchicalSNN

cfg = Config()

train_loader, _ = get_dataloaders(
    batch_size=cfg.batch_size,
    time_steps=cfg.time_steps,
)

spikes, labels = next(iter(train_loader))

print("Original:", spikes.shape)

spikes = spikes.permute(1, 0, 2)

print("Permuted:", spikes.shape)

model = HierarchicalSNN(
    n_input=cfg.n_input,
    n_hidden1=cfg.hidden1,
    n_hidden2=cfg.hidden2,
    n_classes=cfg.n_classes,
)

logits = model(spikes)

print("Logits:", logits.shape)

print("Labels:", labels.shape)