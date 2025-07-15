import torch
import torch.nn as nn
from torchvision import transforms, models
import torch.nn.functional as F

class SemanticFPN(nn.Module):
    def __init__(self, num_channels=1):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # FPN lateral connections
        self.lat_layer4 = nn.Conv2d(2048, 256, kernel_size=1)
        self.lat_layer3 = nn.Conv2d(1024, 256, kernel_size=1)
        self.lat_layer2 = nn.Conv2d(512, 256, kernel_size=1)
        self.lat_layer1 = nn.Conv2d(256, 256, kernel_size=1)

        # FPN smooth layers
        self.smooth4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.smooth3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.smooth2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.smooth1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Feature fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256*4, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Segmentation head
        self.seg_head = self._make_seg_head(256, num_channels)
    
    def _make_seg_head(self, in_channels, out_channels, dropout=0.1):
        return nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(64, out_channels, 1)
        )
    
    def forward(self, x):
        _, _, H, W = x.size()
        x = self.conv1(x)
        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        # FPN Top-down
        pre_smooth4 = self.lat_layer4(c5)
        pre_smooth3 = self.lat_layer3(c4)
        pre_smooth2 = self.lat_layer2(c3)
        pre_smooth1 = self.lat_layer1(c2)

        p5 = self.smooth4(pre_smooth4)
        p4 = self.smooth3(pre_smooth3 + F.interpolate(pre_smooth4, scale_factor=2, mode='nearest'))
        p3 = self.smooth2(pre_smooth2 + F.interpolate(pre_smooth3, scale_factor=2, mode='nearest'))
        p2 = self.smooth1(pre_smooth1 + F.interpolate(pre_smooth2, scale_factor=2, mode='nearest'))
        
        # Feature concatenation
        p3_up = F.interpolate(p3, scale_factor=2, mode='nearest')
        p4_up = F.interpolate(p4, scale_factor=4, mode='nearest')
        p5_up = F.interpolate(p5, scale_factor=8, mode='nearest')
        
        fused = torch.cat([p2, p3_up, p4_up, p5_up], dim=1)
        fused = self.fusion_conv(fused)
        
        # Final prediction
        pred = self.seg_head(fused)
        return torch.sigmoid(F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=True))