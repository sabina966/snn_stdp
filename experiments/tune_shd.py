# experiments/tune_shd.py

import sys
import os
sys.path.insert(0, 'D:/Sabina/Projects/snn_stdp')

import torch
import tonic
import matplotlib.pyplot as plt
import numpy as np
from network.unsupervised_snn import UnsupervisedSNN

# ----- КАНВЕРТАЦЫЯ -----
def convert_events_to_spikes(events, time_steps=200, sensor_size=700):
    spikes = torch.zeros(time_steps, sensor_size)
    t_min = events['t'].min()
    t_max = events['t'].max()
    for event in events:
        t_norm = (event['t'] - t_min) / (t_max - t_min)
        t_idx = min(int(t_norm * time_steps), time_steps - 1)
        channel = event['x']
        spikes[t_idx, channel] = 1.0
    return spikes

# ----- ХУТКІ ТЭСТ -----
def quick_test(params, sample_count=500):
    print(f"\n--- Testing: {params} ---")
    try:
        # Загрузка даных
        train_dataset = tonic.datasets.SHD(save_to='./data', train=True)
        test_dataset = tonic.datasets.SHD(save_to='./data', train=False)

        # Стварэнне сеткі
        network = UnsupervisedSNN(
            n_input=700,
            n_hidden=params['n_hidden'],
            neuron_params={'tau_m': 20.0, 'v_th': -50.0, 'dt': 1.0, 'tau_adapt': 100.0},
            stdp_params={
                'a_plus': params['a_plus'],
                'a_minus': params['a_minus'],
                'tau_plus': 20.0,
                'tau_minus': 20.0,
                'w_min': 0.0,
                'w_max': 1.0
            },
            homeo_params={
                'target_rate': params['target_rate'],
                'tau_homeo': 5000.0,
                'homeo_strength': 0.02,
                'min_homeo_factor': 0.5,
                'max_homeo_factor': 2.0
            }
        )

        # Навучанне
        neuron_responses = {i: [] for i in range(params['n_hidden'])}
        for idx in range(sample_count):
            events, label = train_dataset[idx]
            spikes = convert_events_to_spikes(events)
            _, winner = network.forward(spikes)
            neuron_responses[winner.item()].append(label)

        # Мапінг нейронаў
        neuron_to_digit = {}
        for neuron in range(params['n_hidden']):
            responses = neuron_responses[neuron]
            if responses:
                digit = max(set(responses), key=responses.count)
                neuron_to_digit[neuron] = digit

        # Тэставанне
        correct = 0
        total = 200
        for idx in range(total):
            events, true_label = test_dataset[idx]
            spikes = convert_events_to_spikes(events)
            _, winner = network.forward(spikes, record=False)
            predicted = neuron_to_digit.get(winner.item(), -1)
            if predicted == true_label:
                correct += 1

        accuracy = 100 * correct / total
        print(f"Result: {accuracy:.2f}%")
        return accuracy

    except Exception as e:
        print(f"ERROR: {e}")
        return None

# ----- СПІС ПАРАМЕТРАЎ ДЛЯ ТЭСТАВАННЯ -----
configs = [
    {'n_hidden': 200, 'a_plus': 0.01, 'a_minus': 0.012, 'target_rate': 12.5},

]

# ----- ЗАПУСК -----
print("=" * 60)
print("TUNING STDP PARAMETERS")
print("=" * 60)

results = []
for params in configs:
    acc = quick_test(params)
    results.append((params, acc))

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for params, acc in results:
    if acc is not None:
        print(f"Params: {params} → {acc:.2f}%")
    else:
        print(f"Params: {params} → ERROR")