#include <iostream>
#include "tensor/tensor.hpp"

int main() {
    std::cout << "--- 架构重构后的极简计算图测试 ---" << std::endl;
    
    // 1. 模拟模型预测值
    tensor::Tensor preds({2, 2});
    preds.fill(0.8f); 
    
    // 2. 模拟真实标签
    tensor::Tensor targets({2, 2});
    targets.fill(0.1f);
    
    std::cout << "Predictions:\n" << preds << "\n" << std::endl;
    std::cout << "Targets:\n" << targets << "\n" << std::endl;

    // 3. 极其优雅的数学表达式！
    // 包含了：减法 operator-，乘法 operator*，以及求均值 mean()
    tensor::Tensor error = preds - targets;
    tensor::Tensor squared_error = error * error; 
    float mse_loss = squared_error.mean();
    
    std::cout << "Squared Error Matrix:\n" << squared_error << "\n" << std::endl;
    std::cout << "Final MSE Loss: " << mse_loss << std::endl;

    return 0;
}