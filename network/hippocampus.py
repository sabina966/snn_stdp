
"""
Модель гиппокампа для обработки выходов SNN сети
"""

import torch
import torch.nn as nn

class Hippocampus(nn.Module):
    """
    Упрощённая модель гиппокампа:
    - Энторинальная кора (вход)
    - CA3 (рекуррентная обработка)
    - CA1 (выход)
    """
    
    def __init__(self, input_size, hidden_size=128, output_size=64):
        super().__init__()
        
        # Энторинальная кора
        self.ec = nn.Linear(input_size, hidden_size)
        
        # CA3 (рекуррентная память)
        self.ca3 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        
        # CA1 (выход)
        self.ca1 = nn.Linear(hidden_size, output_size)
        
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, x, return_hidden=False):
        # Энторинальная кора
        ec_out = self.relu(self.ec(x))
        ec_out = self.dropout(ec_out)
        
        # CA3 (временная обработка)
        # Добавляем временное измерение если нужно
        if ec_out.dim() == 2:
            ec_out = ec_out.unsqueeze(1)
        ca3_out, hidden = self.ca3(ec_out)
        ca3_out = ca3_out.squeeze(1)
        
        # CA1 (выход)
        ca1_out = self.ca1(ca3_out)
        
        if return_hidden:
            return ca1_out, hidden
        return ca1_out
