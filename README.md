# 基于c++实现的PPO算法
+ 包含一个actor网络和一个critic网络，每个网络的前向传播函数均为二层线性神经网络。
+ 其中actor使用基于高斯分布采样的方式获取action，也即action为连续空间。
+ 采用了Suitable credit技巧，以及基于clip的相关系数计算。
+ 主函数位于main.cpp中。

