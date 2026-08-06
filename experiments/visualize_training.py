"""
Visualization of SNN training results.

Usage:
    python visualize_training.py checkpoints/run_name
"""

import os
import json
import sys

import matplotlib.pyplot as plt


def load_history(run_dir):

    path = os.path.join(
        run_dir,
        "history.json"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No history.json found in {run_dir}"
        )

    with open(path, "r") as f:
        return json.load(f)



def plot_loss(history, save_path):

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train",
        marker="o"
    )

    plt.plot(
        epochs,
        history["test_loss"],
        label="Test",
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    plt.legend()
    plt.grid()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



def plot_accuracy(history, save_path):

    epochs = range(
        1,
        len(history["train_acc"]) + 1
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        epochs,
        history["train_acc"],
        label="Train",
        marker="o"
    )

    plt.plot(
        epochs,
        history["test_acc"],
        label="Test",
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(
        "Classification Accuracy"
    )

    plt.legend()
    plt.grid()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



def plot_firing_rate(history, save_path):

    epochs = range(
        1,
        len(history["rate1"]) + 1
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        epochs,
        history["rate1"],
        label="Layer 1",
        marker="o"
    )

    plt.plot(
        epochs,
        history["rate2"],
        label="Layer 2",
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Firing rate")

    plt.title(
        "Layer Activity"
    )

    plt.legend()
    plt.grid()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



def main():

    if len(sys.argv) < 2:
        print(
            "Usage:"
        )
        print(
            "python visualize_training.py checkpoints/run_name"
        )
        return


    run_dir = sys.argv[1]

    print(
        "Loading:",
        run_dir
    )


    history = load_history(
        run_dir
    )


    plot_loss(
        history,
        os.path.join(
            run_dir,
            "loss.png"
        )
    )

    plot_accuracy(
        history,
        os.path.join(
            run_dir,
            "accuracy.png"
        )
    )

    plot_firing_rate(
        history,
        os.path.join(
            run_dir,
            "firing_rate.png"
        )
    )


    print(
        "Plots saved in:"
    )
    print(run_dir)



if __name__ == "__main__":
    main()