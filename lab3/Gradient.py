import torch
import numpy as np
import shutup
import matplotlib.pyplot as plt
import random

shutup.please()
random.seed(42)

class Gradient(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = torch.nn.Linear(in_features=2, out_features=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.feature(x))

class TrainingHandler(object):
    def __init__(self):
        self.x = []
        self.y = []
        self.test_x = []
        self.test_y = []
        self.half_data_size = 400
        self.data_size = 800

    def generate_data(self):
        for i in range(self.half_data_size):
            tmp = []
            tmp.append(random.uniform(-10, 5))
            tmp.append(random.uniform(-10, 5))
            self.y.append([0.])
            self.x.append(tmp)
        for i in range(self.half_data_size):
            tmp = []
            tmp.append(random.uniform(5, 20))
            tmp.append(random.uniform(5, 20))
            self.y.append([1.])
            self.x.append(tmp)
        for i in range(self.half_data_size):
            tmp = []
            tmp.append(random.uniform(15, 120))
            tmp.append(random.uniform(15, 120))
            self.test_x.append(tmp)
            self.test_y.append([1.])
        for i in range(self.half_data_size):
            tmp = []
            tmp.append(random.uniform(-120, 0))
            tmp.append(random.uniform(-110, 3))
            self.test_x.append(tmp)
            self.test_y.append([0.])
        self.x = torch.tensor(self.x)
        self.y = torch.tensor(self.y)
        self.test_x = torch.tensor(self.test_x)
        self.test_y = torch.tensor(self.test_y)

    def training_data_visualizer(self):
        plt.figure()
        for i in range(self.data_size):
            if(self.y[i] == [1.]):
                plt.scatter(self.x[i][0], self.x[i][1], s=10, marker='*', c='r')
            else:
                plt.scatter(self.x[i][0], self.x[i][1], s=10, marker='x', c='b')
        plt.title("Pre-training Visualization")
        plt.show()

def main():
    model = Gradient()
    train = TrainingHandler()
    train.generate_data()
    #train.training_data_visualizer()

    loss = torch.nn.BCELoss()
    lr = 0.001
    iter_time = 100
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.7)
    optimizer.zero_grad()
    for epoch in range(iter_time):
        y_pred = model(train.x)
        my_loss = loss(y_pred, train.y)
        my_loss.backward()
        optimizer.step()
        print("epoch: %d, loss: %f" % (epoch , my_loss))

if __name__ == "__main__":
    main()
