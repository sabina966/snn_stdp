import torch

from network.hierarchical_snn import HierarchicalSNN

model = HierarchicalSNN()

x = torch.randint(
    0,
    2,
    (200, 1, 700)
).float()

logits = model(x)

print(logits.shape)