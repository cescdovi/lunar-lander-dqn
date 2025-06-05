import torch
from torch import nn
import torch.nn.functional as F
class DQN(nn.Module):
    def __init__(self, 
                 input_dim: int,    
                 output_dim: int):
        
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
