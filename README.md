# AI-Math-Experiment
Python code of experiment from Harbin Institute of Technology's Essential Mathematics for AI.

哈工大2023人工智能数学基础实验课Python代码。

有问题欢迎发issue。

## 说在前面

本repo里的内容可能没有完全覆盖到实际的实验要求，还请谅解。

克隆到本地：

```bash
git clone https://github.com/coder109/AI-Math-Experiment
```

安装一些依赖库：

```bash
pip install -r requirements.txt
```

在`lab2`下创建文件夹`mnist`，将`mnist_train.csv`文件放在该文件夹下。该文件请访问这里下载：https://github.com/phoebetronic/mnist

## LAB1

本LAB有三个要求：

1. 用RANSAC和最小二乘法尝试拟合一条直线。
2. 用RANSAC和最小二乘法尝试拟合一条曲线。
3. 假设给定函数,从中生成n个点，添加随机噪声,然后用拟合、回归和神经网络的方法来求解模型, 使得模型误差最小!
4. 添加外点，查看拟合的效果。

这个共有3个文件，作用如下：

- `Fitting.py`，具体的函数实现。需要注意的是，在本文件中，我加入了添加外点的功能，若不需要该功能，请自行修改`generateNoises`函数。
- `CurveFitting.py`，拟合曲线的主函数。
- `Straight.py`，拟合直线的主函数。
- `数学基础实验1.ipynb`，第三个问题的实现。

## LAB2

本实验要求为，通过`PCA`和`RPCA`对`MNIST`数据集进行分类。

在本例中，文件作用如下：

1. `mnist`文件夹，及其下面的`mnist_train.csv`文件，为`MNIST`数据集文件。
2. `PCA.py`，解决问题主函数。
3. `RPCA.py`，鲁棒主成分分析的相关代码，感谢https://github.com/dganguli/robust-pca

