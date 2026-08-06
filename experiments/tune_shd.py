# experiments/tune_shd.py

import sys
import os
sys.path.insert(0, 'D:/Sabina/Projects/snn_stdp')

import torch
import tonic
import matplotlib.pyplot as plt
import numpy as np
from legacy.unsupervised_snn import HierarchicalUnsupervisedSNN

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

        # Стварэнне двухслойнай сеткі з галавой-класіфікатарам (для суррага́тнага градыента)
        network = HierarchicalUnsupervisedSNN(
            n_input=700,
            n_hidden_1=params['n_hidden_1'],
            n_hidden_2=params['n_hidden_2'],
            n_classes=20,
            neuron_params={'tau_m': 20.0, 'v_th': -50.0, 'dt': 1.0, 'tau_adapt': 100.0},
            stdp_params_1={
                'a_plus': params['a_plus_1'],
                'a_minus': params['a_minus_1'],
                'a3_plus': params.get('a3_plus_1', params['a_plus_1'] * 0.5),
                'a3_minus': params.get('a3_minus_1', params['a_minus_1'] * 0.5),
                'tau_pre_slow': params.get('tau_pre_slow_1', 100.0),
                'tau_post_slow': params.get('tau_post_slow_1', 100.0),
                'tau_plus': 20.0,
                'tau_minus': 20.0,
                'w_min': 0.0,
                'w_max': 1.0
            },
            stdp_params_2={
                'a_plus': params['a_plus_2'],
                'a_minus': params['a_minus_2'],
                'a3_plus': params.get('a3_plus_2', params['a_plus_2'] * 0.5),
                'a3_minus': params.get('a3_minus_2', params['a_minus_2'] * 0.5),
                'tau_pre_slow': params.get('tau_pre_slow_2', 100.0),
                'tau_post_slow': params.get('tau_post_slow_2', 100.0),
                'tau_plus': 20.0,
                'tau_minus': 20.0,
                'w_min': 0.0,
                'w_max': 1.0
            },
            homeo_params_1={
                'target_rate': params['target_rate'],
                'tau_homeo': 5000.0,
                'homeo_strength': 0.02,
                'min_homeo_factor': 0.5,
                'max_homeo_factor': 2.0
            },
            homeo_params_2={
                'target_rate': params['target_rate'],
                'tau_homeo': 5000.0,
                'homeo_strength': 0.02,
                'min_homeo_factor': 0.5,
                'max_homeo_factor': 2.0
            }
        )

        # Навучанне: першая фаза — бесперагляднае STDP
        neuron_responses = {i: [] for i in range(params['n_hidden_2'])}
        for idx in range(sample_count):
            events, label = train_dataset[idx]
            spikes = convert_events_to_spikes(events)
            _, _, winner = network.forward(spikes, stdp=True)
            neuron_responses[winner.item()].append(label)

        # Короткая supervised фаза з суррага́тным градыентам: 3 эпохі па ~200 крокаў
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        sup_steps = min(200, len(train_dataset))
        sup_epochs = 3
        for epoch in range(sup_epochs):
            for idx in range(sup_steps):
                events, label = train_dataset[idx]
                spikes = convert_events_to_spikes(events)
                # атрымліваем logits, выключаючы STDP падчас supervised крокаў
                hidden1, hidden2, winner, logits = network.forward(spikes, stdp=False, return_logits=True)
                loss = criterion(logits.unsqueeze(0), torch.tensor([label], dtype=torch.long))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"   Supervised epoch {epoch+1}/{sup_epochs} completed")

        # Тэставанне (supervised eval)
        correct = 0
        total = min(200, len(test_dataset))
        for idx in range(total):
            events, true_label = test_dataset[idx]
            spikes = convert_events_to_spikes(events)
            hidden1, hidden2, winner, logits = network.forward(spikes, stdp=False, return_logits=True)
            _, predicted = torch.max(logits, 0)
            if predicted.item() == true_label:
                correct += 1

        accuracy = 100 * correct / total
        print(f"Result (supervised eval): {accuracy:.2f}%")
        return accuracy

    except Exception as e:
        print(f"ERROR: {e}")
        return None

# ----- СПІС ПАРАМЕТРАЎ ДЛЯ ТЭСТАВАННЯ -----
configs = [
    {
        'n_hidden_1': 200,
        'n_hidden_2': 150,
        'a_plus_1': 0.09,
        'a_minus_1': 0.019,
        'a_plus_2': 0.02,
        'a_minus_2': 0.06,
        # Triplet params for layer1
        'a3_plus_1': 0.3,
        'a3_minus_1': 0.2,
        'tau_pre_slow_1': 200.0,
        'tau_post_slow_1': 200.0,
        # Triplet params for layer2
        'a3_plus_2': 0.2,
        'a3_minus_2': 0.03,
        'tau_pre_slow_2': 200.0,
        'tau_post_slow_2': 200.0,
        'target_rate': 13.5,
    },
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