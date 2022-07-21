import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF

def DCAR(in_channels, out_channels):    # Double Conv And ReLU
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels,  kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class UNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNET, self).__init__()

        # Encoder - Contracting Path
        self.en1 = DCAR(in_channels, 64)
        self.en2 = DCAR(64, 128)
        self.en3 = DCAR(128, 256)
        self.en4 = DCAR(256, 512)
        self.en5 = DCAR(512, 1024)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Up-conv
        self.upconv54 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.upconv43 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.upconv32 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.upconv21 = nn.ConvTranspose2d(128, 64, 2, 2)

        # Decoder - Expansive Path
        self.dec4 = DCAR(1024, 512)
        self.dec3 = DCAR(512, 256)
        self.dec2 = DCAR(256, 128)
        self.dec1 = DCAR(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):

        # Encoder
        e1 = self.en1(x)                # concat with d1
        e2 = self.en2(self.pool(e1))    # concat with d2
        e3 = self.en3(self.pool(e2))    # concat with d3
        e4 = self.en4(self.pool(e3))    # concat with d4
        e5 = self.en5(self.pool(e4))

        # Decoder
        d4_1 = self.upconv54(e5)
        d4_2 = torch.cat([e4, d4_1], dim=1)
        d4_3 = self.dec4(d4_2)
        d3_1 = self.upconv43(d4_3)
        d3_2 = torch.cat([e3, d3_1], dim=1)
        d3_3 = self.dec3(d3_2)
        d2_1 = self.upconv32(d3_3)
        d2_2 = torch.cat([e2, d2_1], dim=1)
        d2_3 = self.dec2(d2_2)
        d1_1 = self.upconv21(d2_3)
        d1_2 = torch.cat([e1, d1_1], dim=1)
        d1_3 = self.dec1(d1_2)

        out = self.final(d1_3)

        return out