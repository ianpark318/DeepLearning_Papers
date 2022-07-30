import torch
import torch.nn as nn

def Double_Conv_Block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    return block

def Triple_Conv_Block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    return block

class VGGNet16(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(VGGNet16, self).__init__()

        self.conv_layer = nn.Sequential(
            Double_Conv_Block(in_channels, 64),
            Double_Conv_Block(64, 128),
            Triple_Conv_Block(128, 256),
            Triple_Conv_Block(256, 512),
            Triple_Conv_Block(512, 512)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

model = VGGNet16(in_channels=3, num_classes=10)
print(model)
