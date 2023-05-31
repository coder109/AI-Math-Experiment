from matplotlib import pyplot as plt
import random
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import array

random.seed(42) # 42 is the answer of everything!

def generateNoises(number_size: int) -> list:
    result = []
    for i in range(number_size):
        if i % 10 == 0:
            result.append(random.random()*10)
        else:
            result.append(random.random())
    return result

# parameters = [a, b], which represents y = a*x + b
def generatePointsFromLine(number_size: int, parameters: list):
    x_cordinate = []
    y_cordinate = []
    for i in range(number_size):
        x = i - 10
        x_cordinate.append(x)
        y_cordinate.append(x * parameters[0] + parameters[1])
    return x_cordinate, y_cordinate

# parameters = [a, b, c], which represents y = a*x^2 + b*x + c
def generatePointsFromCurve(number_size: int, parameters: list):
    x_cordinate = []
    y_cordinate = []
    for i in range(number_size):
        x = i - 10
        x_cordinate.append(x)
        y_cordinate.append(x*x*parameters[0] + x*parameters[1] + parameters[2])
    return x_cordinate, y_cordinate

# LSQ for quadratic equation
def curveFitting(x: list, y: list) -> list:
    parameters = np.polyfit(x, y, 2)
    parameters = np.poly1d(parameters)
    return [parameters[0], parameters[1], parameters[2]]

# LSQ for a straight line
def linearFitting(x: list, y: list) -> list:
    size = len(x)
    sum_xy, sum_x, sum_y, sum_x2 = 0, 0, 0, 0
    for i in range(size):
        sum_xy += x[i] * y[i]
        sum_y += y[i]
        sum_x += x[i]
        sum_x2 += x[i] * x[i]
    x_average = sum_x / size
    y_average = sum_y / size
    result_k = (size*sum_xy - sum_x*sum_y) / (size*sum_x2 - sum_x*sum_x)
    result_b = y_average - x_average*result_k
    return [result_k, result_b]

# RANSAC for a straight line
def RANSAC(x: list, y: list) -> list:
    iter_time = 5000
    epsilon = 3
    threshold = 0.5
    result_k, result_b = 0, 0
    max_inner_point_number = 0

    for i in range(iter_time):
        temp_k = (y[len(x)-1] - y[0]) / (x[len(x)-1] - x[0])
        temp_b = y[0] - temp_k*x[0]
        inner_point_counter = 0
        for cord in range(len(x)):
            estimate_y = temp_k * x[cord] + temp_b
            if abs(y[cord] - estimate_y) < epsilon:
                inner_point_counter += 1
            if inner_point_counter > threshold * len(x):
                break
            if inner_point_counter > max_inner_point_number:
                result_k = temp_k
                result_b = temp_b
                max_inner_point_number = inner_point_counter
    return [result_k, result_b]

# RANSAC for quadratic equation
# Modified from:
# https://charleshsliao.wordpress.com/2017/06/16/ransac-and-nonlinear-regression-in-python/
def RANSAC_curve(x: list, y: list):
    x = np.array(x)
    x = x.reshape(-1,1)
    quadratic = PolynomialFeatures(degree=2)
    quadratic.fit(x)
    x = quadratic.transform(x)

    lr = LinearRegression()
    lr.fit(x, y)
    y_predict = lr.predict(x)
    return x, y_predict

def visualizer(x_original: list, y_original: list, function, title: str, function_type: int):
    point_number = len(x_original)
    figure = plt.figure()
    plt.scatter(x_original, y_original, color='r', s=10)
    if function_type == 0:
        y_test = [elem*function[0]+function[1] for elem in x_original]
    if function_type == 1:
        y_test = [elem*function[0]*elem+elem*function[1]+function[2] for elem in x_original]
    plt.plot(x_original, y_test, color='g')
    plt.title(title)
    plt.show()

# Cannot figure out how to extract the goddamn function from RANSAC_curve
# So I just write a new visualizer!
def visualizer_points(x_original: list, y_original: list, y_new: list, title: str):
    point_number = len(x_original)
    figure = plt.figure()
    plt.scatter(x_original, y_original, color='r', s=10)
    plt.plot(x_original, y_new, color='g', lw=2)
    plt.title(title)
    plt.show()
