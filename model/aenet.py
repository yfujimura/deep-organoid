import torch
from torch import nn
import torch.nn.functional as F

class AENet(nn.Module):
    
    def __init__(self, encoder, num_frame=1):
        super().__init__()
        
        self.upconv3 = self._upConvLayer(2048, 1024, 3)
        self.iconv3 = self._convLayer(2048, 1024, 3)
        
        self.upconv2 = self._upConvLayer(1024, 512, 3)
        self.iconv2 = self._convLayer(1024, 512, 3)
        
        self.upconv1 = self._upConvLayer(512, 256, 3)
        self.iconv1 = self._convLayer(512, 256, 3)
        
        self.upconv0 = self._upConvLayer(256, 64, 3)
        self.iconv0 = self._convLayer(64, 64, 3)
        
        self.outconv = nn.Conv2d(64, 1, 3, padding=1)
        
        
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
                    
        self.encoder = encoder
        
    def forward(self, x):
        if x.shape[1] == 1:
            conv1, conv2, conv3, self.conv4 = self.encoder(torch.cat((x,x,x), 1))
        # conv1: 256 x H/4 x W/4
        # conv2: 512 x H/8 x W/8
        # conv3: 1024 x H/16 x W/16
        # conv4: 2048 x H/32 x W/32
        
        upconv3 = self.upconv3(self.conv4) # 1024 x H/16 x W/16
        iconv3 = self.iconv3(torch.cat((upconv3, conv3), 1)) # 1024 x H/16 x W/16
        
        upconv2 = self.upconv2(iconv3) # 512 x H/8 x W/8
        iconv2 = self.iconv2(torch.cat((upconv2, conv2), 1)) # 512 x H/8 x W/8
        
        upconv1 = self.upconv1(iconv2) # 256 x H/4 x W/4
        iconv1 = self.iconv1(torch.cat((upconv1, conv1), 1)) # 256 x W/4 x H/4
                             
        upconv0 = self.upconv0(iconv1) # 64 x W/2 x H/2
        iconv0 = self.iconv0(upconv0) # 64 x H/2 x W/2
        
        out = torch.sigmoid(self.outconv(iconv0)) # 1 x H/2 x W/2
        
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


class Encoder(nn.Module):

    def __init__(self, original_model, num_features = 2048):
        super().__init__()        
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool

        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
       

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)

        return x_block1, x_block2, x_block3, x_block4
    
    