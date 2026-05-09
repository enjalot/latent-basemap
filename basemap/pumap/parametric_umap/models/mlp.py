import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with flexible architecture.
    
    Args:
        input_dim (int): Dimension of the input features.
        hidden_dim (int): Dimension of the hidden layers (embedding dimension).
        output_dim (int): Dimension of the output layer.
        num_layers (int): Number of hidden layers.
        use_batchnorm (bool): If True, includes Batch Normalization after each linear layer.
        use_dropout (bool): If True, includes Dropout after each activation function.
        dropout_prob (float): Probability of an element to be zeroed. Default: 0.5.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, 
                 use_batchnorm=False, use_dropout=False, dropout_prob=0.5):
        super(MLP, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        for i in range(num_layers):
            # Linear layer
            layers.append(nn.Linear(in_dim, hidden_dim))
            
            # Batch Normalization (optional)
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation function
            layers.append(nn.ReLU())
            
            # Dropout (optional)
            if use_dropout:
                layers.append(nn.Dropout(dropout_prob))
            
            # Update input dimension for next layer
            in_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(in_dim, output_dim))
        
        # Optionally, you can add an activation function for the output
        # For example, use Sigmoid for binary classification or Softmax for multi-class
        # layers.append(nn.Softmax(dim=1))
        
        # Combine all layers into a Sequential module
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass of the MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        return self.model(x)


class ResidualBottleneckMLP(nn.Module):
    """Bottleneck MLP with residual blocks in the hidden bottleneck."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        neck_dim = hidden_dim * 3 // 4
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.down = nn.Sequential(nn.Linear(hidden_dim, neck_dim), nn.ReLU())
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(neck_dim, neck_dim), nn.ReLU())
            for _ in range(max(num_layers - 1, 0))
        ])
        self.up = nn.Sequential(nn.Linear(neck_dim, hidden_dim), nn.ReLU())
        self.proj_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.proj_in(x))
        x = self.down(x)
        for block in self.blocks:
            x = x + block(x)
        x = self.up(x)
        return self.proj_out(x)
