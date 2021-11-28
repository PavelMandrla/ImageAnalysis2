import torch
import torch.nn as nn


class VGG_block(nn.Module):
    def __init__(self, count, in_channels, out_channels, batch_norm=True):
        super(VGG_block, self).__init__()

        self.layers = []
        for i in range(count):
            self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if batch_norm:
                self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        #self.net = nn.Sequential(*layers)  # TODO - is this OK?

    #def forward(self, x):
        #return self.net(x)


class VGG(nn.Module):
    def __init__(self, cfg, in_channels=3, batch_norm=True):
        super(VGG, self).__init__()
        self.blocks = []

        if isinstance(cfg, str):
            cfg = VGG.cfg[cfg]
        for count, out_channels in cfg:
            self.blocks.append(VGG_block(count, in_channels, out_channels, batch_norm=batch_norm))
            in_channels = out_channels

        #self.net = nn.Sequential(*blocks)

    #def forward(self, x):
        #return self.net(x)

    cfg = {
        '11': [(1, 64), (1, 128), (2, 256), (2,512), (2, 512)],
        '13': [(2, 64), (2, 128), (2, 256), (2,512), (2, 512)],
        '16': [(2, 64), (2, 128), (3, 256), (3,512), (3, 512)],
        '19': [(2, 64), (2, 128), (4, 256), (4,512), (4, 512)],
    }
