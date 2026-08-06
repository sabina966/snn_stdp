from datasets import SHDDataset

dataset = SHDDataset()

print(len(dataset))

spikes, label = dataset[0]

print(spikes.shape)

print(label)

print(spikes.sum())