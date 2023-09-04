import torch
import torch.nn as nn
from constants import *

class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        mp1 = 10
        filters_1 = 32
        kernel_size_1 = 5
        filters_2 = 64
        kernel_size_2 = 5
        hidden = 100
        self.conv1 = torch.nn.Conv2d(3, filters_1, kernel_size_1)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=mp1, stride=mp1)
        self.conv3 = torch.nn.Conv2d(filters_1, filters_2, kernel_size_2)
        #self.dropout1 = torch.nn.Dropout(0.2)
        self.fc1 = torch.nn.Linear(filters_2 * (int((SCREEN_WIDTH - kernel_size_1 + 1)/mp1) - kernel_size_2 + 1) * (int((SCREEN_HEIGHT - kernel_size_1 + 1)/mp1)  - kernel_size_2 + 1), hidden)
        #self.dropout2 = torch.nn.Dropout(0.2)
        self.out = torch.nn.Linear(hidden, 3)


    def forward(self, x):
        first = self.conv1(x)
        m1 = nn.ReLU()
        first = m1(first)
        first = self.maxpool1(first)
        #print(f"first is {first}")
        second = self.conv3(first)
        second = m1(second)
        second = second.view(second.size(0),-1)
        #second = self.dropout1(second)
        #print(f"second is {second}")
        third = self.fc1(second)
        third = m1(third)
        #third = self.dropout2(third)
        #print(f"third is {third}")
        output = self.out(third)
        m2 = nn.Sigmoid()
        return m2(output)

