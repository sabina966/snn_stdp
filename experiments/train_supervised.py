# experiments/supervised_hierarchical_shd.py

import sys
sys.path.insert(0, 'D:/Sabina/Projects/snn_stdp')

import torch
import torch.nn as nn
import torch.optim as optim
import tonic
import matplotlib.pyplot as plt
import numpy as np
from legacy.unsupervised_snn import HierarchicalUnsupervisedSNN

# ----- ПЕРАЎТВАРЭННЕ СПАЙКАЎ -----
def convert_events_to_spikes(events, time_steps=200, sensor_size=(700, 1, 1)):
    spikes = torch.zeros(time_steps, sensor_size[0])
    t_min = events['t'].min()
    t_max = events['t'].max()
    for event in events:
        t_norm = (event['t'] - t_min) / (t_max - t_min)
        t_idx = min(int(t_norm * time_steps), time_steps - 1)
        channel = event['x']
        spikes[t_idx, channel] = 1.0
    return spikes

# ----- НАВУЧАННЕ З НАСТАЎНІКАМ -----
def main():
    print("=" * 60)
    print("SUPERVISED TWO-LAYER SNN (SURROGATE GRADIENT)")
    print("Layer 1: 200 neurons, Layer 2: 100 neurons")
    print("Training with labels (supervised)!")
    print("Hot start: STDP + Surrogate Gradient")
    print("=" * 60)

    # 1. Загрузка даных
    print("\n1. Loading SHD dataset...")
    train_dataset = tonic.datasets.SHD(save_to='./data', train=True)
    test_dataset = tonic.datasets.SHD(save_to='./data', train=False)
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")

    # 2. Стварэнне сеткі з класіфікатарам
    print("\n2. Creating supervised SNN...")
    network = HierarchicalUnsupervisedSNN(
        n_input=700,
        n_hidden_1=200,
        n_hidden_2=100,
        n_classes=20,
        neuron_params={'tau_m': 20.0, 'v_th': -50.0, 'dt': 1.0, 'tau_adapt': 50.0},
        stdp_params_1={
            'a_plus': 0.01,
            'a_minus': 0.012,
            'tau_plus': 20.0,
            'tau_minus': 20.0,
            'w_min': 0.0,
            'w_max': 1.0,
        },
        stdp_params_2={
            'a_plus': 0.005,
            'a_minus': 0.006,
            'tau_plus': 20.0,
            'tau_minus': 20.0,
            'w_min': 0.0,
            'w_max': 1.0,
        },
        homeo_params_1={
            'target_rate': 12.5,
            'tau_homeo': 5000.0,
            'homeo_strength': 0.02,
            'min_homeo_factor': 0.5,
            'max_homeo_factor': 2.0,
        },
        homeo_params_2={
            'target_rate': 12.5,
            'tau_homeo': 5000.0,
            'homeo_strength': 0.02,
            'min_homeo_factor': 0.5,
            'max_homeo_factor': 2.0,
        },
    )
    print("   Input: 700 → Layer1: 200 → Layer2: 100 → Classes: 20")
    print("   Learning: Hot start (STDP → Surrogate Gradient)")

    # ----- 2a. ГАРАЧЫ СТАРТ: STDP (без настаўніка) -----
    print("\n2a. Hot start: STDP pre-training...")
    stdp_steps = 500
    for idx in range(stdp_steps):
        events, _ = train_dataset[idx]
        spikes = convert_events_to_spikes(events)
        network.forward(spikes, stdp=True, return_logits=False)
        if (idx + 1) % 100 == 0:
            print(f"   STDP step {idx + 1}/{stdp_steps}")
    print(f"   STDP pre-training complete!")

    # ----- 3. Навучанне з настаўнікам (сурогатны градыент) -----
    print("\n3. Supervised training (surrogate gradient)...")
    optimizer = optim.Adam(network.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    n_epochs = 10
    samples_per_epoch = min(2000, len(train_dataset))

    for epoch in range(n_epochs):
        total_loss = 0
        correct = 0

        for idx in range(samples_per_epoch):
            events, label = train_dataset[idx]
            spikes = convert_events_to_spikes(events)

            # Forward pass (stdp=False, return_logits=True)
            _, _, _, logits = network.forward(
                spikes, 
                stdp=False,      # адключаем STDP
                return_logits=True
            )

            loss = criterion(logits.unsqueeze(0), torch.tensor([label], dtype=torch.long))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits, 0)
            if predicted.item() == label:
                correct += 1

            if (idx + 1) % 500 == 0:
                acc = 100 * correct / (idx + 1)
                print(f"   Epoch {epoch+1}: {idx+1}/{samples_per_epoch}, Loss: {total_loss/(idx+1):.4f}, Acc: {acc:.2f}%")

        epoch_acc = 100 * correct / samples_per_epoch
        print(f"   Epoch {epoch+1} completed! Train accuracy: {epoch_acc:.2f}%")

    # 4. Тэставанне
    print("\n4. Testing...")
    correct = 0
    test_samples = 1000

    for idx in range(test_samples):
        events, true_label = test_dataset[idx]
        spikes = convert_events_to_spikes(events)

        _, _, _, logits = network.forward(spikes, stdp=False, return_logits=True)
        _, predicted = torch.max(logits, 0)

        if predicted.item() == true_label:
            correct += 1

        if (idx + 1) % 200 == 0:
            print(f"   Tested {idx + 1}/{test_samples}")

    accuracy = 100 * correct / test_samples
    print(f"\n{'='*60}")
    print(f"SUPERVISED ACCURACY: {accuracy:.2f}%")
    print(f"{'='*60}")

    # 5. Візуалізацыя ваг
    print("\n5. Visualizing learned weights...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    weights1 = network.layer1.weights.detach().numpy()
    weights2 = network.layer2.weights.detach().numpy()

    ax = axes[0]
    im = ax.imshow(weights1, aspect='auto', cmap='coolwarm', vmin=0, vmax=1)
    ax.set_xlabel('Layer1 Neurons')
    ax.set_ylabel('Input Channels')
    ax.set_title('Layer1 Weights (Supervised)')
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    im = ax.imshow(weights2, aspect='auto', cmap='coolwarm', vmin=0, vmax=1)
    ax.set_xlabel('Layer2 Neurons')
    ax.set_ylabel('Layer1 Neurons')
    ax.set_title('Layer2 Weights (Supervised)')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('supervised_hierarchical_weights.png', dpi=150)
    print("   Saved visualization to 'supervised_hierarchical_weights.png'")
    plt.show()

if __name__ == '__main__':
    main()