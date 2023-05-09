import torch
from torch import nn
import torch.nn.functional as F

class AbsNet(nn.Module):
    
    def __init__(self, in_channels):
        super().__init__()
        
        self.conv1 = self._convLayer(in_channels, 256, 1)
        self.conv2 = self._convLayer(256, 256, 1)
        self.conv3 = nn.Conv2d(256, 2, 1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        
    def forward(self, x):
        x = F.adaptive_max_pool2d(x, (1,1))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x) # N x 2 x 1 x 1
        max_value = x[:,0,:,:].unsqueeze(1)
        min_value = x[:,1,:,:].unsqueeze(1)
        min_value = min_value * 0 + 0.007843138
        return max_value, min_value
    
    def _convLayer(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=int((kernel_size-1)/2), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )