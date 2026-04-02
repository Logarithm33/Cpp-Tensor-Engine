#include <iostream>
#include <iomanip>
#include "tensor/tensor.hpp"
#include "ops/math_ops.hpp"
#include "ops/act_ops.hpp"
#include "nn/linear.hpp"

int main() {
    std::cout << "--- 神经网络极简训练循环测试 (Gradient Descent) ---" << std::endl;
    
    // 1. 准备训练数据
    // 输入 X: 4 个样本，每个样本 2 个特征
    tensor::Tensor X({4, 2});
    X.data()[0] = 0.0f; X.data()[1] = 0.0f;
    X.data()[2] = 0.0f; X.data()[3] = 1.0f;
    X.data()[4] = 1.0f; X.data()[5] = 0.0f;
    X.data()[6] = 1.0f; X.data()[7] = 1.0f;

    // 目标 Y: 我们想让网络学习简单的 "AND" 逻辑门规律
    // 只有当两个输入都是 1 时，输出才是 1，其他全是 0
    tensor::Tensor Target({4, 1});
    Target.data()[0] = 0.0f;
    Target.data()[1] = 0.0f;
    Target.data()[2] = 0.0f;
    Target.data()[3] = 1.0f;

    // 2. 实例化神经网络层
    // 输入特征 2，输出特征 1
    nn::Linear layer(2, 1);

    // 设置学习率 (Learning Rate)
    float lr = 0.1f;

    std::cout << "开始训练..." << std::endl;

    // 3. 训练循环 (Epochs)
    for (int epoch = 1; epoch <= 500; ++epoch) {
        
        // --- 第一步：清空上一次的梯度 (optimizer.zero_grad) ---
        layer.weight().zero_grad();
        layer.bias().zero_grad();

        // --- 第二步：前向传播 (Forward Pass) ---
        tensor::Tensor Y_pred = layer.forward(X);
        
        // 计算 MSE Loss
        tensor::Tensor diff = Y_pred - Target;
        tensor::Tensor loss = ops::mean(diff * diff);

        // --- 第三步：反向传播 (Backward Pass) ---
        loss.backward();

        // --- 第四步：梯度下降更新参数 (optimizer.step) ---
        // W = W - lr * dW
        float* w_data = layer.weight().data();
        const float* w_grad = layer.weight().grad();
        for (size_t i = 0; i < layer.weight().size(); ++i) {
            w_data[i] -= lr * w_grad[i];
        }

        // b = b - lr * db
        float* b_data = layer.bias().data();
        const float* b_grad = layer.bias().grad();
        for (size_t i = 0; i < layer.bias().size(); ++i) {
            b_data[i] -= lr * b_grad[i];
        }

        // 每 10 轮打印一次 Loss
        if (epoch % 10 == 0 || epoch == 1) {
            std::cout << "Epoch " << std::setw(2) << epoch 
                      << " | Loss: " << std::fixed << std::setprecision(6) << loss.data()[0] << std::endl;
        }
    }

    std::cout << "\n训练结束！测试模型预测结果：" << std::endl;
    tensor::Tensor final_pred = layer.forward(X);
    std::cout << final_pred << std::endl;

    return 0;
}