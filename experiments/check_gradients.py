import torch

from network.hierarchical_snn import HierarchicalSNN

model = HierarchicalSNN()

x = torch.randint(0, 2, (200, 1, 700)).float()

target = torch.tensor([3])

criterion = torch.nn.CrossEntropyLoss()

logits = model(x)

loss = criterion(logits, target)

loss.backward()

print("Loss:", loss.item())

print("Layer1 grad:",
      model.layer1.weights.grad is not None)

print("Layer2 grad:",
      model.layer2.weights.grad is not None)

print("Classifier grad:",
      model.classifier.network[0].weight.grad is not None)