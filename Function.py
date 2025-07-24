import torch
import torch.nn as nn

class GELU(nn.Module):  # Eredita da nn.Module
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Applica la funzione GELU
        return 0.5 * x * (1 + torch.tanh((2 / torch.pi) ** 0.5 * (x + 0.044715 * x ** 3)))