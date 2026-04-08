#include <iostream>
#include <iomanip>
#include "tensor/tensor.hpp"
#include "ops/math_ops.hpp"
#include "ops/act_ops.hpp"
#include "nn/linear.hpp"

int main() {
    std::cout << "--- 终极分类器测试：Softmax + 交叉熵 (Cross-Entropy) ---" << std::endl;
    
    // 1. 准备训练数据 (Batch Size = 4, Features = 2)
    // 依然是 AND 逻辑门的数据
    tensor::Tensor X({4, 2});
    X.data()[0] = 0.0f; X.data()[1] = 0.0f;
    X.data()[2] = 0.0f; X.data()[3] = 1.0f;
    X.data()[4] = 1.0f; X.data()[5] = 0.0f;
    X.data()[6] = 1.0f; X.data()[7] = 1.0f;

    // 2. 准备 One-Hot 标签 (Batch Size = 4, Classes = 2)
    // [1.0, 0.0] 代表 0， [0.0, 1.0] 代表 1
    tensor::Tensor Target({4, 2});
    Target.data()[0] = 1.0f; Target.data()[1] = 0.0f; // 0 AND 0 = 0
    Target.data()[2] = 1.0f; Target.data()[3] = 0.0f; // 0 AND 1 = 0
    Target.data()[4] = 1.0f; Target.data()[5] = 0.0f; // 1 AND 0 = 0
    Target.data()[6] = 0.0f; Target.data()[7] = 1.0f; // 1 AND 1 = 1

    // 3. 实例化神经网络层
    // 输入特征 2，输出类别数为 2
    nn::Linear layer(2, 2);

    float lr = 0.5f; // 交叉熵的梯度非常稳定，我们可以用较大的学习率加速收敛！

    std::cout << "开始训练..." << std::endl;

    // 4. 训练循环 (Epochs)
    for (int epoch = 1; epoch <= 100; ++epoch) { // 只需 100 轮！
        
        // --- 梯度清零 ---
        layer.weight().zero_grad();
        layer.bias().zero_grad();

        // --- 前向传播：计算 Logits ---
        tensor::Tensor logits = layer.forward(X);
        
        // --- 算误差：使用融合算子！内部包含了 Softmax 和 CE ---
        tensor::Tensor loss = ops::CrossEntropyWithLogits(logits, Target);

        // --- 反向传播 ---
        loss.backward();

        // --- 梯度下降更新参数 ---
        float* w_data = layer.weight().data();
        const float* w_grad = layer.weight().grad();
        for (size_t i = 0; i < layer.weight().size(); ++i) {
            w_data[i] -= lr * w_grad[i];
        }

        float* b_data = layer.bias().data();
        const float* b_grad = layer.bias().grad();
        for (size_t i = 0; i < layer.bias().size(); ++i) {
            b_data[i] -= lr * b_grad[i];
        }

        // 打印 Loss
        if (epoch % 10 == 0 || epoch == 1) {
            std::cout << "Epoch " << std::setw(3) << epoch 
                      << " | Loss: " << std::fixed << std::setprecision(6) << loss.data()[0] << std::endl;
        }
    }

    // 5. 见证奇迹：推理测试！
   // 5. 见证奇迹：推理测试！
    std::cout << "\n训练结束！测试模型预测结果：" << std::endl;
    
    tensor::Tensor final_logits = layer.forward(X);
    
    // 【关键修复】：剥离计算图！安全进入 Softmax！
    tensor::Tensor final_probs = ops::Softmax(final_logits.detach());
    
    std::cout << "预测概率分布 (左列为类别0概率，右列为类别1概率):" << std::endl;
    std::cout << final_probs << std::endl;

    return 0;
}