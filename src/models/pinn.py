# pinn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import fem_torch

class PINN(nn.Module):
    def __init__(self, input_dim_coords, input_dim_config, output_dim, hidden_dims=[128, 128, 128]):
        """
        Initializes the PINN model.

        Args:
            input_dim_coords (int): Dimension of the input coordinates/density grid.
            input_dim_config (int): Dimension of the input configuration (e.g., forces).
            output_dim (int): Dimension of the output density grid.
            hidden_dims (list of int, optional): Sizes of hidden layers. Defaults to [128, 128, 128].
        """
        super(PINN, self).__init__()
        layers = []
        prev_dim = input_dim_coords + input_dim_config  # Total input dimension after concatenation

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # Ensures output densities are between 0 and 1

        self.network = nn.Sequential(*layers)
    
    def forward(self, coords, config):
        """
        Forward pass of the PINN.

        Args:
            coords (torch.Tensor): Tensor representing the initial density grid.
                                    Shape: (batch_size, input_dim_coords)
            config (torch.Tensor): Tensor representing the forces configuration.
                                   Shape: (batch_size, input_dim_config)

        Returns:
            torch.Tensor: Refined density grid.
                          Shape: (batch_size, output_dim)
        """
        # Ensure inputs are of expected dimensions
        if coords.dim() != 2:
            raise ValueError(f"coords should be a 2D tensor, got {coords.dim()}D tensor instead.")
        if config.dim() != 2:
            raise ValueError(f"config should be a 2D tensor, got {config.dim()}D tensor instead.")

        # Concatenate coords and config along the feature dimension
        input_tensor = torch.cat([coords, config], dim=1)  # Shape: (batch_size, input_dim_coords + input_dim_config)
        # print(f"coords device is {coords.device}")
        # Forward pass through the network
        density = self.network(input_tensor)  # Shape: (batch_size, output_dim)

        return density


# pinn.py

def train_pinn(pinn_model, initial_design, forces, device, opt, epochs=10, lr=1e-3):
    """
    Trains the PINN to refine the initial design based on forces to minimize compliance.

    Args:
        pinn_model (PINN): The PINN model to train.
        initial_design (torch.Tensor): Initial density grid. Shape: (batch_size, input_dim_coords)
        forces (torch.Tensor): Forces configuration. Shape: (batch_size, input_dim_config)
        material_properties (dict): Material properties for compliance calculation.
        epochs (int, optional): Number of training epochs. Defaults to 10.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        device (str or torch.device, optional): Device to train on. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Refined density grid. Shape: (batch_size, output_dim)
    """
    pinn_model.to(device)
    pinn_model.train()
    optimizer = opt #optim.Adam(pinn_model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        refined_design = pinn_model(initial_design, forces)  # Shape: (batch_size, output_dim)
        normals, _ , density = fem_torch.mbb_beam_1()
        forces = forces
        args = fem_torch.get_args(normals, forces, density)

        compliance, mass = fem_torch.compliance_and_constraint(refined_design, args)  # Shape: (batch_size,)
        # print(f"Compliance requires grad: {compliance.requires_grad}")
        # print(f"Mass constraint requires grad: {mass.requires_grad}")
        lambda_mass = 500.0
        mass_constraint = torch.relu(mass - args['density'])
        loss = (compliance + lambda_mass * mass_constraint).mean()  # Minimize average compliance
        # print(f"Loss device: {loss.device}")
        # print(f"Loss requires grad: {loss.requires_grad}")
        assert loss.requires_grad, "Loss does not require gradients!"
        loss.backward()
        optimizer.step()
        
        if torch.isnan(loss):
            print(f"NaN detected in loss at epoch {epoch}.")
            break
    pinn_model.eval()
    return refined_design.detach()
