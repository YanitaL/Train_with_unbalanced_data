"""
Convolutional neuron network to train unbalanced data
"""

import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(512)
        self.max = nn.MaxPool2d(2, stride=2)
        self.linear = nn.Linear(8192,10)
        self.relu = nn.ReLU()

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        N = x.shape[0]
        C = x.shape[1]
        y = self.conv(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.max(y)
        y = self.batchnorm(y)
        y = self.conv3(y)
        y = self.relu(y)
        y = self.conv4(y)
        y = self.relu(y)
        y = self.max(y)
        y = self.batchnorm2(y)
        y = self.conv5(y)
        y = self.relu(y)
        y = self.conv6(y)
        y = self.relu(y)
        y = self.max(y)
        y2 = self.batchnorm3(y)
        y2_f = y2.view(N, -1)
        m = y2_f.shape[1]
        outs = torch.zeros(N, 10)
        if torch.cuda.is_available():
            outs = outs.cuda()
            y2_f = y2_f.cuda()
        for i in range(N):
            outs[i, :] += self.linear(y2_f[i, :])
        return outs
