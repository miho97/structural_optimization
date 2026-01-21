import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

class SurrogateModel(nn.Module):
    """
    A CNN-based surrogate model to predict structural compliance from a state
    represented as a multi-channel image. The input state (e.g., shape (5, 7, 7))
    is normalized using training data statistics. The model outputs a single scalar
    compliance value.
    """
    def __init__(self, grid_channels=5, grid_height=7, grid_width=7, normalize=True):
        super(SurrogateModel, self).__init__()
        self.grid_channels = grid_channels
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.normalize = normalize
        
        # Register buffers for normalization with shape (1, grid_channels, 1, 1)
        self.register_buffer("state_mean", torch.zeros(1, grid_channels, 1, 1))
        self.register_buffer("state_std", torch.ones(1, grid_channels, 1, 1))
        
        # Define a CNN backbone.
        self.cnn = nn.Sequential(
            nn.Conv2d(grid_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((grid_height // 2, grid_width // 2))
        )
        cnn_out_h = grid_height // 2
        cnn_out_w = grid_width // 2
        cnn_output_size = 64 * cnn_out_h * cnn_out_w
        
        # Fully-connected layers for regression.
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Predicts a single compliance value.
        )
        
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, grid_channels, grid_height, grid_width)
        Returns:
            Tensor of shape (batch_size, 1) representing predicted compliance.
        """
        if self.normalize:
            # Normalize input using stored mean and std.
            x = (x - self.state_mean) / (self.state_std + 1e-8)
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        compliance = self.fc(features)
        return compliance

    def compute_normalization_params(self, states_tensor):
        """
        Compute per-channel mean and std from the training states.
        Args:
            states_tensor: Tensor of shape (N, grid_channels, grid_height, grid_width)
        """
        # Compute mean and std over the batch, height, and width dimensions.
        mean = states_tensor.mean(dim=[0, 2, 3], keepdim=True)  # shape: (1, grid_channels, 1, 1)
        std = states_tensor.std(dim=[0, 2, 3], keepdim=True)    # shape: (1, grid_channels, 1, 1)
        self.state_mean.copy_(mean)
        self.state_std.copy_(std)

    def train_model(self, dataset, batch_size=32, epochs=10, learning_rate=1e-3, device='cpu', grad_clip=0.5):
        """
        Train the surrogate model.
        
        Args:
            dataset: dict or tuple with keys 'states' and 'compliances'. States should
                     have shape (N, grid_channels, grid_height, grid_width) and compliances (N,).
            batch_size: Mini-batch size.
            epochs: Number of epochs.
            learning_rate: Learning rate.
            device: 'cpu' or 'cuda'.
            grad_clip: Maximum norm for gradient clipping.
            
        Returns:
            loss_history: List of average losses per epoch.
            grad_norm_history: List of average gradient norms per epoch.
        """
        if isinstance(dataset, dict):
            states = dataset['states']
            compliances = dataset['compliances']
        else:
            states, compliances = dataset
        
        # Convert to tensors.
        states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
        compliances_tensor = torch.tensor(compliances, dtype=torch.float32, device=device)
        if compliances_tensor.dim() == 1:
            compliances_tensor = compliances_tensor.unsqueeze(1)
        
        # Compute normalization parameters on the full dataset.
        self.compute_normalization_params(states_tensor)
        
        ds = TensorDataset(states_tensor, compliances_tensor)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        self.to(device)
        self.train()
        
        loss_history = []
        grad_norm_history = []
        
        print("Starting training of the surrogate model...")
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_grad_norms = []
            for batch_states, batch_compliances in loader:
                optimizer.zero_grad()
                outputs = self.forward(batch_states)
                loss = criterion(outputs, batch_compliances)
                loss.backward()
                
                # Compute gradient norm.
                total_norm = 0.0
                for p in self.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                epoch_grad_norms.append(total_norm)
                
                # Apply gradient clipping.
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                
                optimizer.step()
                epoch_loss += loss.item() * batch_states.size(0)
            avg_loss = epoch_loss / len(ds)
            avg_grad_norm = np.mean(epoch_grad_norms)
            loss_history.append(avg_loss)
            grad_norm_history.append(avg_grad_norm)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Avg Grad Norm: {avg_grad_norm:.4f}")
        print("Training completed.")
        return loss_history, grad_norm_history

    def predict(self, x, device='cpu'):
        """
        Predict compliance given a state or batch of states.
        Args:
            x: Tensor or numpy array of shape (N, grid_channels, grid_height, grid_width).
            device: 'cpu' or 'cuda'.
        Returns:
            Numpy array of predictions.
        """
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=device)
        with torch.no_grad():
            prediction = self.forward(x)
        return prediction.cpu().numpy()

    def plot_training_metrics(self, loss_history, grad_norm_history, save_path=None):
        """
        Plot training loss and gradient norm over epochs.
        Args:
            loss_history: List of loss values per epoch.
            grad_norm_history: List of gradient norm values per epoch.
            save_path: If provided, save the plot to this path.
        """
        epochs = range(1, len(loss_history) + 1)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        color = 'tab:blue'
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color=color)
        ax1.plot(epochs, loss_history, color=color, label="Loss")
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel("Gradient Norm", color=color)
        ax2.plot(epochs, grad_norm_history, color=color, label="Gradient Norm")
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
