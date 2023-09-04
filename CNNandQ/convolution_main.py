from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn.metrics as metrics
import numpy as np
from torchvision import datasets, transforms
from torchinfo import summary
from convolution import Conv
import os

def train(model, device, train_loader, train_outputs, optimizer, epoch):
    model.train()
    
    for batch_idx, (data,target) in enumerate(zip(train_loader, train_outputs)):
        data, target = data.type(torch.FloatTensor).to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.MSELoss()
        l = loss(output, target)
        #print(data)
        #print(f"output is {output}")
        #print(f"target is {target}")
        l.backward()
        #print(l)
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), l.item()))

def test(model, device, test_loader, test_outputs):
    model.eval()
    test_loss = 0
    loss = nn.MSELoss()
    with torch.no_grad():
        for data, target in zip(test_loader, test_outputs):
            data, target = data.type(torch.FloatTensor).to(device), target.to(device)
            output = model(data)
            #print(test_loss)
            #print(data)
            #print(f"output is {output}")
            #print(f"target is {target}")
            test_loss += loss(output, target).item()
    print(f"total loss is {test_loss}")

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}\n'.format(
        test_loss))


def main():
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = torch.load('tensors.pt')
    output = torch.load('outputs.pt')
    #print(output)
    print(data.size())
    train_size = int(0.65 * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size], generator = torch.Generator().manual_seed(42))
    train_output, test_output = torch.utils.data.random_split(output, [train_size, test_size], generator = torch.Generator().manual_seed(42))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=False)
    train_outputs = torch.utils.data.DataLoader(train_output, batch_size=8, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)
    test_outputs = torch.utils.data.DataLoader(test_output, batch_size=8, shuffle=False)
    # choose network architecture
    net = Conv().to(device)
    summary(net)
    if list(net.parameters()):
        # use SGD optimizer
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.1)

        # training and testing loop
        for epoch in range(1, 100 + 1):
            print(f"epoch is {epoch}")
            train(net, device, train_loader, train_outputs , optimizer=optimizer, epoch=epoch)
            test(net, device, test_loader, test_outputs)
    local_dir = os.path.dirname(__file__)
    file_path = os.path.join(local_dir, "conv.pth")
    torch.save(net.state_dict(), file_path)
if __name__ == '__main__':
    main()