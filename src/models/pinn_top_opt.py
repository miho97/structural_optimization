
import torch
import torch.nn as nn

class DensityPINN(nn.Module):
    def __init__(self, width, height, hidden_dim=64, config_dim=4):

        super(DensityPINN, self).__init__()
        self.width = width
        self.height = height
        self.num_elements = width * height
        self.hidden_dim = hidden_dim
        self.config_dim = config_dim

        self.net = nn.Sequential(
            nn.Linear(2 + config_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  
        )

    def forward(self, coords, config):

        input_tensor = torch.cat([coords, config], dim=1)  # Shape: (num_elements, 2 + config_dim)
        density = self.net(input_tensor).squeeze(-1)
        return density
