import torch
from torch import nn
import torch.nn.functional as F

class Unet(nn.Module):
    
    def __init__(self, num_frame=1):
        super().__init__()
        
        self.conv1 = self._downConvLayer(num_frame, 32, 7)
        self.conv2 = self._downConvLayer(32, 64, 5)
        self.conv3 = self._downConvLayer(64, 128, 3)
        self.conv4 = self._downConvLayer(128, 256, 3)
        self.conv5 = self._downConvLayer(256, 512, 3)
        
        self.upconv4 = self._upConvLayer(512, 256, 3)
        self.iconv4 = self._convLayer(512, 256, 3)
        
        self.upconv3 = self._upConvLayer(256, 128, 3)
        self.iconv3 = self._convLayer(256, 128, 3)
        
        self.upconv2 = self._upConvLayer(128, 64, 3)
        self.iconv2 = self._convLayer(128, 64, 3)
        
        self.upconv1 = self._upConvLayer(64, 32, 3)
        self.iconv1 = self._convLayer(64, 32, 3)
        
        self.upconv0 = self._upConvLayer(32, 32, 3)
        
        self.outconv = nn.Conv2d(32, 1, 3, padding=1)
        
        
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
        conv1 = self.conv1(x)  # 32 x H/2 x W/2
        conv2 = self.conv2(conv1)  # 64 x H/4 x W/4
        conv3 = self.conv3(conv2)  # 128 x H/8 x W/8
        conv4 = self.conv4(conv3)  # 256 x H/16 x W/16
        conv5 = self.conv5(conv4)  # 512 x H/32 x W/32
        
        upconv4 = self.upconv4(conv5) # 256 x H/16 x W/16
        iconv4 = self.iconv4(torch.cat((upconv4, conv4), 1)) # 256 x H/16 x W/16
        
        upconv3 = self.upconv3(iconv4) # 128 x H/8 x W/8
        iconv3 = self.iconv3(torch.cat((upconv3, conv3), 1)) # 128 x H/8 x W/8
        
        upconv2 = self.upconv2(iconv3) # 64 x H/4 x W/4
        iconv2 = self.iconv2(torch.cat((upconv2, conv2), 1)) # 64 x W/4 x H/4
                             
        upconv1 = self.upconv1(iconv2) # 32 x W/2 x H/2
        iconv1 = self.iconv1(torch.cat((upconv1, conv1), 1)) # 32 x H/2 x W/2
        
        upconv0 = self.upconv0(iconv1) # 32 x H x W
        out = torch.sigmoid(self.outconv(upconv0)) # 1 x H x W
        
        return out
    
    def _convLayer(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=int((kernel_size-1)/2), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        
    def _downConvLayer(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=int((kernel_size-1)/2), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=int((kernel_size-1)/2), stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def _upConvLayer(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=int((kernel_size-1)/2), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )