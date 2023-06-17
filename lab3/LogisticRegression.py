import matplotlib.pyplot as plt
import shutup
import numpy as np
import seaborn as sns
import pandas as pd
import random

# Ignore warnings
shutup.please()

# Define some parameters
theta_origin = 4
random.seed(42)

# Define some function we need to use
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class Logistic(object):
    def __init__(self):
        self.data_size = 600
        self.half_data_size = 300
        self.iter_time = 1000
        self.x = []
        self.y = []
        self.w = np.ones((3, 1))
        self.lr = 0.001
        self.loss_list = []

    def generate_data(self):
        for i in range(self.half_data_size):
            tmp = []
            tmp.append(random.uniform(-10, 5))
            tmp.append(random.uniform(-10, 5))
            tmp.append(random.uniform(-10, 5))
            self.y.append([0.])
            self.x.append(tmp)
        for i in range(self.half_data_size):
            tmp = []
            tmp.append(random.uniform(5, 20))
            tmp.append(random.uniform(5, 20))
            tmp.append(random.uniform(5, 20))
            self.y.append([1.])
            self.x.append(tmp)
        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def training_data_visualizer(self):
        plt.figure()
        for i in range(self.data_size):
            if(self.y[i] == [1.]):
                plt.scatter(self.x[i][0], self.x[i][1], s=10, marker='*', c='r')
            else:
                plt.scatter(self.x[i][0], self.x[i][1], s=10, marker='x', c='b')
        plt.title("Pre-training Visualization")
        plt.show()
    
    def sigmoid(self):
        eta = -np.dot(self.x, self.w)
        H = np.exp(eta)
        H = 1.0 / (1.0 + H)
        return H

    def Newton(self):
        for i in range(self.iter_time):
            s = self.sigmoid()
            loss = np.log(s) * self.y + (1-self.y) * np.log(1-s)
            loss = -np.sum(loss) / self.data_size
            self.loss_list.append(loss)
            dw = self.x.T.dot(s-self.y) / self.data_size
            self.w -= self.lr * self.w
            print("iter: %d, loss: %f" % (i, loss))
        print(self.w)
    
    def show_loss_curve(self):
        x = np.arange(0, self.iter_time, 1)
        plt.plot(x, np.array(self.loss_list), label='Curve')
        plt.xlabel("iteration time")
        plt.ylabel("loss value")
        plt.title("Loss Curve")
        plt.show()

    def inference(self):
        test_value = [random.uniform(-10, 20), random.uniform(-10, 20), random.uniform(-10, 20)]
        test_y = 1 / (1 + np.exp(-np.dot(test_value, self.w)))
        print(test_y)

    def show_classification(self):
        fig = plt.figure()
        for i in range(self.data_size):
            if(self.y[i] == [1.]):
                plt.scatter(self.x[i][0], self.x[i][1], s=10, marker='*', c='r')
            else:
                plt.scatter(self.x[i][0], self.x[i][1], s=10, marker='x', c='b')
        x = np.arange(-20, 20, 0.02)
        y = (-self.w[0] - self.w[1] * x) / self.w[2]
        plt.scatter(x, y, s=10, c='g')
        plt.title("Classfication Visualizer")
        plt.show()

def main():
    train_handler = Logistic()
    train_handler.generate_data()
    train_handler.training_data_visualizer()
    train_handler.Newton()
    train_handler.show_loss_curve()
    train_handler.inference()
    train_handler.show_classification()

if __name__ == "__main__":
    main()
