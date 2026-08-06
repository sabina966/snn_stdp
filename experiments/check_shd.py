import h5py


file = h5py.File(
    "data/SHD/shd_train.h5",
    "r"
)


def show(name, obj):
    print(name, "->", type(obj))


file.visititems(show)