from datasets.shd import SHDDataset


dataset = SHDDataset(
    path="data/SHD",
    train=True
)


print("Samples:", len(dataset))


x, y = dataset[0]


print("Shape:", x.shape)
print("Label:", y)
print("Spike count:", x.sum())