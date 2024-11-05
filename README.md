# 基于C++实现的PPO算法Demo

本项目是一个使用C++实现的PPO（Proximal Policy Optimization）算法演示。PPO是一种流行的强化学习算法，适用于连续动作空间的问题。

## 特点

- **网络结构**：包含一个actor网络和一个critic网络，每个网络的前向传播函数均为二层线性神经网络。
- **动作空间**：actor使用基于高斯分布采样的方式获取action，即action为连续空间。
- **技术应用**：采用了Suitable credit技巧，以及基于clip的相关系数计算。
- **代码结构**：主函数位于`main.cpp`中。

## 组件

### tools.h

- **Matrix**：实现了矩阵的加减乘除以及其他常用操作。
- **AdamOptimizer**：一个adam优化器。
- **LinearLayer**：一个线性层。
- **激活函数**：包括sigmoid, relu, softmax操作。
- **损失函数**：包括MSE和crossentropy损失函数。

### main.cpp

- **Env**：模拟强化学习中的环境，提供相应的接口。
- **PPO**：PPO模型定义。
- **训练环境**：包含模型定义以及训练的主环境。

