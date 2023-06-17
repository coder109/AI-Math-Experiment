import pandas as pd
import numpy as np
import shutup
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
import time

shutup.please()

# Define some hyper-parameters
batch_size = 128
device = torch.device('cuda')

# Define a class for SGD
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2))
        self.dense = nn.Sequential(nn.Linear(14*14*128, 1024),
                                   nn.ReLU(),
                                   nn.Dropout(p = 0.5),
                                   nn.Linear(1024, 10))
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 14*14*128)
        x = self.dense(x)
        return x

def main():
    '''
    There are some format-related issues here,
    but I am too tired to fix them.
    '''
    train = datasets.MNIST(root="./mnist",
                       train=True,
                       transform=transforms.ToTensor(),
                       download=True)
    train = DataLoader(train,
                   shuffle=True,
                   batch_size=batch_size)
    test = datasets.MNIST(root="./mnist",
                      train=False,
                      transform=transforms.ToTensor(),
                      download=True)
    test = DataLoader(test,
                  shuffle=True,
                  batch_size=batch_size)

    model = Model().to(device)
    loss = nn.CrossEntropyLoss()
    lr = 0.01
    iter_time = 10
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8)
    print("Using SGD with a trainging epoch 10:")
    start = time.time()
    for epoch in range(iter_time):
        total = 0
        correct = 0
        for data in train:
            inputs, facts = data
            inputs = inputs.to(device)
            facts = facts.to(device)
            optimizer.zero_grad()
    
            for tester in test:
                test_inputs, test_facts = tester
                test_inputs = test_inputs.to(device)
                test_facts = test_facts.to(device)
                test_output = model(test_inputs)
                _, predicted = torch.max(test_output.data, dim=1)
                total += test_facts.size(0)
                correct += (predicted == test_facts).sum().item()

            output = model(inputs)
            epoch_loss = loss(output, facts)
            epoch_loss.backward()
            optimizer.step()
    
        if epoch == iter_time - 1:
            print("Accuracy: %f" % (correct * 100.0 / total))
    end = time.time()
    print("Time Cost: %fs" % (end - start))

if __name__ == "__main__":
    main()
