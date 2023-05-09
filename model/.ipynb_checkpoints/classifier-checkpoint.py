import torch
from torch import nn
import torch.nn.functional as F

class Classifier(nn.Module):
    
    def __init__(self, latent_size):
        super().__init__()
        
        self.layer = nn.Linear(latent_size, 2)
        
    def forward(self, x):
        x = self.layer(x)
        return x