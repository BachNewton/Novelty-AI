"""
DQN Neural Network - Deep Q-Network architecture for Snake AI.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """
    Deep Q-Network with 3 hidden layers.

    Architecture:
    - Input: 11 features (game state encoding)
    - Hidden 1: 256 neurons + ReLU + Dropout(0.1)
    - Hidden 2: 256 neurons + ReLU + Dropout(0.1)
    - Hidden 3: 128 neurons + ReLU
    - Output: 3 Q-values (one per action: straight, right, left)
    """

    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 256,
        output_size: int = 3
    ):
        """
        Initialize the network.

        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layers
            output_size: Number of output Q-values (actions)
        """
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)

        self.dropout = nn.Dropout(0.1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Q-values tensor of shape (batch_size, output_size)
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


