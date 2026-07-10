import torch
import tonic
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.insert(0, 'D:/Sabina/Projects/snn_stdp')
from network.unsupervised_snn import HierarchicalUnsupervisedSNN


def convert_events_to_spikes(events, time_steps=200, sensor_size=(700, 1, 1)):
    """Convert SHD events into a [time_steps, channels] spike tensor."""
    spikes = torch.zeros(time_steps, sensor_size[0])
    t_min = events['t'].min()
    t_max = events['t'].max()
    for event in events:
        t_norm = (event['t'] - t_min) / (t_max - t_min)
        t_idx = min(int(t_norm * time_steps), time_steps - 1)
        channel = event['x']
        spikes[t_idx, channel] = 1.0
    return spikes


def main():
    print("=" * 60)
    print("TWO-LAYER UNSUPERVISED STDP CLASSIFIER")
    print("Layer 1: 200 neurons, a_plus = 0.01")
    print("Layer 2: 100 neurons, a_plus = 0.005")
    print("No labels used during training!")
    print("=" * 60)

    print("\n1. Loading SHD dataset...")
    train_dataset = tonic.datasets.SHD(save_to='./data', train=True)
    test_dataset = tonic.datasets.SHD(save_to='./data', train=False)
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")

    print("\n2. Creating hierarchical unsupervised SNN...")
    network = HierarchicalUnsupervisedSNN(
        n_input=700,
        n_hidden_1=200,
        n_hidden_2=100,
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
    print("   Input: 700 → Layer1: 200 → Layer2: 100")
    print("   Learning: STDP + lateral inhibition in both layers")

    print("\n3. Unsupervised STDP training...")
    print("   Presenting samples, no labels given...")

    sample_count = 2000
    neuron_responses = {i: [] for i in range(100)}
    for idx in range(sample_count):
        events, label = train_dataset[idx]
        spikes = convert_events_to_spikes(events)
        _, _, winner = network.forward(spikes)
        neuron_responses[winner.item()].append(label)
        if (idx + 1) % 500 == 0:
            print(f"   Processed {idx + 1}/{sample_count} samples")
    print("\n   Training complete!")

    print("\n4. Mapping second-layer neurons to digits...")
    neuron_to_digit = {}
    for neuron in range(100):
        responses = neuron_responses[neuron]
        if responses:
            neuron_to_digit[neuron] = max(set(responses), key=responses.count)

    digit_counts = {d: 0 for d in range(20)}
    for digit in neuron_to_digit.values():
        digit_counts[digit] += 1
    print("   Neuron specialization:")
    for digit in range(20):
        print(f"   Digit {digit:2d} → {digit_counts[digit]} neurons")

    print("\n5. Testing...")
    correct = 0
    test_samples = 1000
    for idx in range(test_samples):
        events, true_label = test_dataset[idx]
        spikes = convert_events_to_spikes(events)
        _, _, winner = network.forward(spikes, record=False)
        predicted = neuron_to_digit.get(winner.item(), -1)
        if predicted == true_label:
            correct += 1
        if (idx + 1) % 200 == 0:
            print(f"   Tested {idx + 1}/{test_samples}")
    accuracy = 100 * correct / test_samples
    print(f"\n{'=' * 60}")
    print(f"HIERARCHICAL UNSUPERVISED ACCURACY: {accuracy:.2f}%")
    print(f"{'=' * 60}")

    print("\n6. Visualizing learned weights...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    weights1 = network.layer1.weights.detach().numpy()
    weights2 = network.layer2.weights.detach().numpy()
    ax = axes[0, 0]
    im = ax.imshow(weights1, aspect='auto', cmap='coolwarm', vmin=0, vmax=1)
    ax.set_xlabel('Layer1 Neurons')
    ax.set_ylabel('Input Channels')
    ax.set_title('Layer1 Weights')
    plt.colorbar(im, ax=ax)
    ax = axes[0, 1]
    im = ax.imshow(weights2, aspect='auto', cmap='coolwarm', vmin=0, vmax=1)
    ax.set_xlabel('Layer2 Neurons')
    ax.set_ylabel('Layer1 Neurons')
    ax.set_title('Layer2 Weights')
    plt.colorbar(im, ax=ax)
    ax = axes[1, 0]
    digits = list(range(20))
    counts = [digit_counts[d] for d in digits]
    ax.bar(digits, counts)
    ax.set_xlabel('Digit Class')
    ax.set_ylabel('Number of Specialized Neurons')
    ax.set_title('Second-layer Neuron Specialization')
    ax = axes[1, 1]
    neuron_activity = [len(neuron_responses[i]) for i in range(100)]
    ax.hist(neuron_activity, bins=20, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Samples Won')
    ax.set_ylabel('Layer2 Neurons')
    ax.set_title('Layer2 Winner Distribution')
    plt.tight_layout()
    plt.savefig('unsupervised_hierarchical_stdp_results.png', dpi=150)
    print("   Saved visualization to 'unsupervised_hierarchical_stdp_results.png'")
    plt.show()


if __name__ == '__main__':
    main()
