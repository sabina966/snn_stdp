import matplotlib.pyplot as plt
import torch
import numpy as np


plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
})


# ---------------------------------------------------
# Synaptic weights visualization
# ---------------------------------------------------

def plot_weights(
    weights,
    save_path,
    title="Размеркаванне сінаптычных вагаў",
):

    weights = weights.detach().cpu().numpy()

    plt.figure(figsize=(10, 6))

    plt.imshow(
        weights,
        aspect="auto",
    )

    plt.colorbar(
        label="Значэнне вагі"
    )

    plt.title(title)

    plt.xlabel(
        "Нейроны наступнага слоя"
    )

    plt.ylabel(
        "Нейроны папярэдняга слоя"
    )

    plt.text(
        0.02,
        0.95,
        f"min = {weights.min():.3f}\nmax = {weights.max():.3f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(alpha=0.5),
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ---------------------------------------------------
# SHD class names
# ---------------------------------------------------

def get_shd_class_names():

    return [
        "EN_0",
        "EN_1",
        "EN_2",
        "EN_3",
        "EN_4",
        "EN_5",
        "EN_6",
        "EN_7",
        "EN_8",
        "EN_9",
        "DE_0",
        "DE_1",
        "DE_2",
        "DE_3",
        "DE_4",
        "DE_5",
        "DE_6",
        "DE_7",
        "DE_8",
        "DE_9",
    ]


# ---------------------------------------------------
# Confusion matrix
# ---------------------------------------------------

def plot_confusion_matrix(
    labels,
    predictions,
    save_path,
    n_classes,
):

    cm = np.zeros(
        (n_classes, n_classes),
        dtype=int,
    )

    labels = labels.cpu().numpy()
    predictions = predictions.cpu().numpy()

    for true, pred in zip(labels, predictions):
        cm[true, pred] += 1


    plt.figure(
        figsize=(12, 10)
    )

    plt.imshow(
        cm,
        aspect="auto",
    )

    plt.colorbar(
        label="Колькасць прыкладаў"
    )


    class_names = get_shd_class_names()


    plt.xticks(
        range(n_classes),
        class_names,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(n_classes),
        class_names,
    )


    plt.xlabel(
        "Прадказаны клас"
    )

    plt.ylabel(
        "Сапраўдны клас"
    )

    plt.title(
        "Матрыца памылак класіфікацыі SHD"
    )


    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

# ---------------------------------------------------
# Digit confusion matrix (EN + DE combined)
# ---------------------------------------------------

def plot_digit_confusion_matrix(
    labels,
    predictions,
    save_path,
):
    """
    Confusion matrix for digits only.

    EN_0 + DE_0 -> 0
    EN_1 + DE_1 -> 1
    ...
    EN_9 + DE_9 -> 9
    """


    cm = np.zeros(
        (10, 10),
        dtype=int,
    )


    labels = labels.cpu().numpy()
    predictions = predictions.cpu().numpy()


    # Пераўтварэнне класаў SHD:
    #
    # 0-9   -> EN digit
    # 10-19 -> DE digit
    #
    # Напрыклад:
    # class 13 -> digit 3

    true_digits = labels % 10
    pred_digits = predictions % 10


    for true, pred in zip(
        true_digits,
        pred_digits
    ):
        cm[true, pred] += 1



    plt.figure(
        figsize=(9, 8)
    )


    plt.imshow(
        cm,
        aspect="auto",
    )


    plt.colorbar(
        label="Колькасць прыкладаў"
    )


    digit_names = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]


    plt.xticks(
        range(10),
        digit_names,
    )


    plt.yticks(
        range(10),
        digit_names,
    )


    plt.xlabel(
        "Прадказаная лічба"
    )


    plt.ylabel(
        "Сапраўдная лічба"
    )


    plt.title(
        "Матрыца памылак распазнавання лічбаў SHD"
    )


    # значэнні ў клетках
    for i in range(10):
        for j in range(10):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=9,
            )


    plt.tight_layout()


    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close()

# ---------------------------------------------------
# Spike raster plot
# ---------------------------------------------------

def plot_raster(
    spikes,
    save_path,
    title="Спайкавая актыўнасць нейронаў",
    max_neurons=50,
):

    """
    Raster plot найбольш актыўных нейронаў.

    spikes:
        Shape [time, batch, neurons]

    """

    # агульная колькасць спайкаў кожнага нейрона
    activity = spikes.sum(
        dim=(0, 1)
    )


    # выбар найбольш актыўных
    top = torch.topk(
        activity,
        k=min(
            max_neurons,
            spikes.shape[2]
        )
    ).indices


    # першы прыклад batch
    sample = spikes[:, 0, top]


    # пазіцыі спайкаў
    time_idx, local_idx = torch.nonzero(
        sample,
        as_tuple=True,
    )


    # арыгінальныя нумары нейронаў
    neuron_idx = top[local_idx]


    plt.figure(
        figsize=(10, 6)
    )


    plt.scatter(
        time_idx.cpu(),
        neuron_idx.cpu(),
        s=8,
    )


    plt.xlabel(
        "Часавы крок"
    )

    plt.ylabel(
        "Нумар нейрона"
    )


    plt.title(
        title +
        "\n(найбольш актыўныя нейроны)"
    )


    plt.grid(
        alpha=0.3
    )


    plt.tight_layout()


    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close()

def plot_metric(
    x,
    curves,
    labels,
    save_path,
    title,
    xlabel,
    ylabel,
):
    """
    Universal plotting function.
    """

    plt.figure(figsize=(8, 5))

    markers = ["o", "s", "^", "d", "x"]

    for values, label, marker in zip(
        curves,
        labels,
        markers,
    ):
        plt.plot(
            x,
            values,
            marker=marker,
            linewidth=2,
            markersize=5,
            label=label,
        )

    plt.title(title)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
    )

    plt.close()

def plot_loss(
    history,
    save_path,
):

    epochs = range(
        1,
        len(history["train_loss"]) + 1,
    )

    plot_metric(
        x=epochs,
        curves=[
            history["train_loss"],
            history["test_loss"],
        ],
        labels=[
            "Навучальная выбарка",
            "Тэставая выбарка",
        ],
        save_path=save_path,
        title="Змяненне функцыі страт",
        xlabel="Эпоха",
        ylabel="Loss",
    )

def plot_accuracy(
    history,
    save_path,
):

    epochs = range(
        1,
        len(history["train_acc"]) + 1,
    )

    train = [
        100 * x
        for x in history["train_acc"]
    ]

    test = [
        100 * x
        for x in history["test_acc"]
    ]

    plot_metric(
        x=epochs,
        curves=[
            train,
            test,
        ],
        labels=[
            "Навучальная выбарка",
            "Тэставая выбарка",
        ],
        save_path=save_path,
        title="Дакладнасць класіфікацыі",
        xlabel="Эпоха",
        ylabel="Accuracy (%)",
    )

def plot_firing_rate(
    history,
    save_path,
):

    epochs = range(
        1,
        len(history["rate1"]) + 1,
    )

    layer1 = [
        100 * x
        for x in history["rate1"]
    ]

    layer2 = [
        100 * x
        for x in history["rate2"]
    ]

    plot_metric(
        x=epochs,
        curves=[
            layer1,
            layer2,
        ],
        labels=[
            "Першы слой",
            "Другі слой",
        ],
        save_path=save_path,
        title="Сярэдняя частата генерацыі спайкаў",
        xlabel="Эпоха",
        ylabel="Firing rate (%)",
    )

def plot_weight_histogram(
    weights,
    save_path,
    title="Размеркаванне сінаптычных ваг",
):
    """
    Plot histogram of synaptic weights.
    """

    weights = weights.detach().cpu().numpy().ravel()

    plt.figure(figsize=(8, 5))

    plt.hist(
        weights,
        bins=40,
    )

    plt.title(title)

    plt.xlabel("Вага сінапса")

    plt.ylabel("Колькасць")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
    )

    plt.close()