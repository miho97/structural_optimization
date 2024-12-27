# pinn_topology_optimization.py

import torch
import torch.nn as nn

class DensityPINN(nn.Module):
    def __init__(self, width, height, hidden_dim=64):

        super(DensityPINN, self).__init__()
        self.width = width
        self.height = height
        self.num_elements = width * height
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  
        )

    def forward(self, coords):

        density = self.net(coords).squeeze(-1)
        return density
