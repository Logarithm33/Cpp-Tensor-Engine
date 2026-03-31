#include <iostream>
#include "tensor/tensor.hpp"
#include "ops/act_ops.hpp"
#include "nn/linear.hpp" // 引入刚写的线性层

int main() {
    std::cout << "--- 面向对象的神经网络模块测试 ---" << std::endl;
    
    // 1. 准备输入数据 X (Batch size = 2, Input features = 3)
    tensor::Tensor X = tensor::Tensor::randn({2, 3});
    std::cout << "Input X:\n" << X << "\n" << std::endl;

    // 2. 实例化两个“层”积木
    // 第一层：输入 3，输出 8
    nn::Linear layer1(3, 8);
    // 第二层：输入 8，输出 4
    nn::Linear layer2(8, 4);

    // 3. 极度干净的流式前向传播！
    std::cout << "--- Forward Pass ---" << std::endl;
    
    tensor::Tensor out1 = layer1.forward(X);
    tensor::Tensor act1 = ops::ReLU(out1); // 注意：上一把你把函数名改成了大写 ReLU
    
    tensor::Tensor out2 = layer2.forward(act1);
    
    std::cout << "Final Output Y:\n" << out2 << std::endl;

    return 0;
}