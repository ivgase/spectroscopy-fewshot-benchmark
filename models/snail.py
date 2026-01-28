import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.snail_resnet import *

class SnailFewShot(nn.Module):
    def __init__(self, shots, queries, use_cuda=True):
        # N-way, K-shot
        super(SnailFewShot, self).__init__()
        self.encoder = ResNet1D(ResidualBlock1D, [2, 2, 2, 2], num_classes=1)
        num_channels = 512 + 1

        num_filters = int(math.ceil(math.log(shots+1, 2)))
        self.attention1 = AttentionBlock(num_channels, 64, 32)
        num_channels += 32
        self.tc1 = TCBlock(num_channels, shots+1, 128)
        num_channels += num_filters * 128
        self.attention2 = AttentionBlock(num_channels, 256, 128)
        num_channels += 128
        self.tc2 = TCBlock(num_channels, shots+1, 128)
        num_channels += num_filters * 128
        self.attention3 = AttentionBlock(num_channels, 512, 256)
        num_channels += 256
        self.fc = nn.Linear(num_channels, 1)
        self.shots = shots
        self.queries = queries
        self.use_cuda = use_cuda

    def forward(self, input, labels, shots=None):
        x = self.encoder(input)
        if shots is None:
            shots = self.shots
        batch_size = int(labels.size()[0] / (shots + 1))
        last_idxs = [(i + 1) * (shots + 1) - 1 for i in range(batch_size)]
        if self.use_cuda:
            labels[last_idxs] = torch.Tensor(np.zeros((batch_size, labels.size()[1]))).cuda()
        else:
            labels[last_idxs] = torch.Tensor(np.zeros((batch_size, labels.size()[1])))
        x = torch.cat((x, labels), 1)
        x = x.view((batch_size, shots + 1, -1))
        x = self.attention1(x)
        x = self.tc1(x)
        x = self.attention2(x)
        x = self.tc2(x)
        x = self.attention3(x)
        x = self.fc(x)
        return x
