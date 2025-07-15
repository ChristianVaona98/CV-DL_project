import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Encoder(nn.Module):
    def __init__(self, input_channel, out_channel, dropout):
        super(Encoder, self).__init__()
        self.conv2d_1 = DoubleConv(input_channel, out_channel)
        self.maxpool = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv2d_1(x)
        p = self.maxpool(x)
        p = self.dropout(p)
        return x, p

class Decoder(nn.Module):
    def __init__(self, input_channel, output_channel, dropout):
        super(Decoder, self).__init__()
        self.conv_t = nn.ConvTranspose2d(input_channel, output_channel, stride=2, kernel_size=2)
        self.conv2d_1 = DoubleConv(output_channel * 2, output_channel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, skip):
        x = self.conv_t(x)
        x = torch.cat([x, skip], dim=1)
        x = self.dropout(x)
        x = self.conv2d_1(x)
        return x

class UNetFromScratch(nn.Module):
    def __init__(self, n_channels=3, out_channels=1, dropout=[0.07, 0.08, 0.09, 0.1]):
        super().__init__()
        self.encoder1 = Encoder(n_channels, 64, dropout[0])
        self.encoder2 = Encoder(64, 128, dropout[1])
        self.encoder3 = Encoder(128, 256, dropout[2])
        self.encoder4 = Encoder(256, 512, dropout[3])        
        self.conv_block = DoubleConv(512, 1024)
        self.decoder1 = Decoder(1024, 512, dropout[3])
        self.decoder2 = Decoder(512, 256, dropout[2])
        self.decoder3 = Decoder(256, 128, dropout[1])
        self.decoder4 = Decoder(128, 64, dropout[0])
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        x1, p1 = self.encoder1(x)
        x2, p2 = self.encoder2(p1)
        x3, p3 = self.encoder3(p2)
        x4, p4 = self.encoder4(p3)
        x5 = self.conv_block(p4)
        x6 = self.decoder1(x5, x4)
        x7 = self.decoder2(x6, x3)
        x8 = self.decoder3(x7, x2)
        x9 = self.decoder4(x8, x1)
        logits = self.outc(x9)
        return torch.sigmoid(logits)