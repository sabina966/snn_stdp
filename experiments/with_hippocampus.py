
"""
Эксперимент: SNN с гиппокампом на SHD
"""
import sys
import importlib

# Очистить кэш модуля network
if 'network.unsupervised_snn' in sys.modules:
    del sys.modules['network.unsupervised_snn']

import os
sys.path.insert(0, 'D:/Sabina/Projects/snn_stdp')
import torch
import torch.utils.data as data
import tonic
from network.unsupervised_snn import UnsupervisedSNN
from network.hippocampus import Hippocampus

def convert_events_to_spikes(events, time_steps=200, sensor_size=700):
    """Конвертация SHD событий в спайки"""
    spikes = torch.zeros(time_steps, sensor_size)
    
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
    print("SNN С ГИППОКАМПОМ И ГОМЕОСТАЗОМ НА SHD ДАТАСЕТЕ")
    print("=" * 60)
    
    # 1. Загрузка данных
    print("\n1. Загрузка SHD датасета...")
    train_dataset = tonic.datasets.SHD(save_to='./data', train=True)
    test_dataset = tonic.datasets.SHD(save_to='./data', train=False)
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    
    # 2. Параметры
    n_input = 700
    n_hidden = 500  
    n_classes = 20
    temporal_window = 90  
    sample_count = 500
    epochs = 3

    # 3. Создание сетей
    print("\n2. Создание SNN с гиппокампом и гомеостазисом...")
    
    # Слуховая кора
    auditory_cortex = UnsupervisedSNN(
        n_input=n_input,
        n_hidden=n_hidden,
        neuron_params={'tau_m': 20.0, 'v_th': -50.0, 'dt': 1.0, 'tau_adapt': 50.0},
        stdp_params={
            'a_plus': 0.02, 
            'a_minus': 0.025,
            'tau_plus': 20.0,      
            'tau_minus': 20.0,     
            'w_min': 0.0,          
            'w_max': 1.0             
            },
        homeo_params={
            'target_rate': 10.0,
            'tau_homeo': 5000.0,
            'homeo_strength': 0.02,
            'min_homeo_factor': 0.5,
            'max_homeo_factor': 2.0
        }
    )
    
    # Гиппокамп
    population_size = temporal_window * n_hidden
    hippocampus = Hippocampus(
        input_size=population_size,
        hidden_size=128,
        output_size=64
    )
    
    # Классификатор
    classifier = torch.nn.Linear(64, n_classes)
    
    print(f"   Слуховая кора: {n_input} → {n_hidden} нейронов")
    print(f"   Популяционный вектор: {population_size}")
    print(f"   Гиппокамп: 128 → 64")
    print(f"   Гомеостаз: target_rate=10 Гц, tau_homeo=5000 мс")
    
    # 4. Обучение
    print("\n3. Обучение...")
    
    optimizer = torch.optim.Adam(
        list(hippocampus.parameters()) + list(classifier.parameters()),
        lr=0.001
    )
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        
        for idx in range(sample_count):
            events, label = train_dataset[idx]
            spikes = convert_events_to_spikes(events)
            
            # Прямой проход через слуховую кору
            hidden_spikes, _ = auditory_cortex.forward(spikes)
            
            # Популяционный вектор (последние temporal_window шагов)
            if hidden_spikes.shape[0] >= temporal_window:
                population = hidden_spikes[-temporal_window:, :].flatten()
            else:
                pad = temporal_window - hidden_spikes.shape[0]
                population = torch.cat([
                    hidden_spikes.flatten(),
                    torch.zeros(pad * hidden_spikes.shape[1])
                ])
            
            # Гиппокамп и классификация
            hippo_out = hippocampus(population.unsqueeze(0))
            logits = classifier(hippo_out.squeeze(0))
            
            loss = criterion(logits.unsqueeze(0), torch.tensor([label], dtype=torch.long))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 0)
            if predicted.item() == label:
                correct += 1
            
            if (idx + 1) % 100 == 0:
                print(f"   Epoch {epoch+1}: {idx+1}/{sample_count}, Loss: {total_loss/(idx+1):.4f}")
        
        acc = 100 * correct / sample_count
        print(f"   Epoch {epoch+1} завершён! Точность на обучении: {acc:.2f}%")
    
    # 5. Тестирование
    print("\n4. Тестирование...")
    correct = 0
    total = 200
    
    for idx in range(total):
        events, true_label = test_dataset[idx]
        spikes = convert_events_to_spikes(events)
        
        hidden_spikes, _ = auditory_cortex.forward(spikes)
        
        if hidden_spikes.shape[0] >= temporal_window:
            population = hidden_spikes[-temporal_window:, :].flatten()
        else:
            pad = temporal_window - hidden_spikes.shape[0]
            population = torch.cat([
                hidden_spikes.flatten(),
                torch.zeros(pad * hidden_spikes.shape[1])
            ])
        
        hippo_out = hippocampus(population.unsqueeze(0))
        logits = classifier(hippo_out.squeeze(0))
        
        _, predicted = torch.max(logits, 0)
        if predicted.item() == true_label:
            correct += 1
        if (idx + 1) % 50 == 0:
            print(f"   Тест: {idx+1}/{total}, Точность: {100*correct/(idx+1):.1f}%")
            
    accuracy = 100 * correct / total
    print(f"\n{'='*60}")
    print(f"ИТОГОВАЯ ТОЧНОСТЬ: {accuracy:.2f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()