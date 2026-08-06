from datasets import get_dataloaders

train_loader, _ = get_dataloaders(batch_size=8)

spikes, labels = next(iter(train_loader))

print(spikes.shape)
print(labels.shape)