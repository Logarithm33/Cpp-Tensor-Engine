#include <iostream>
#include "tensor/tensor.hpp"
#include "ops/math_ops.hpp"
#include "ops/act_ops.hpp"

int main() {
    std::cout << "--- 完整 MSE Loss 反向传播测试 ---" << std::endl;
    
    // 1. 定义数据 (Batch Size = 2, Features = 2)
    tensor::Tensor X({2, 2});
    X.data()[0] = 1.0f; X.data()[1] = -2.0f;
    X.data()[2] = 3.0f; X.data()[3] = 0.5f;

    tensor::Tensor Target({2, 2});
    Target.fill(1.0f); // 目标全是 1.0

    // 2. 定义可学习的权重 W (开启求导！)
    tensor::Tensor W = tensor::Tensor::randn({2, 2}, 0.0f, 1.0f, true);
    std::cout << "Initial Weight W:\n" << W << "\n" << std::endl;

    // 3. 前向传播：搭积木！
    // Y = ReLU(X) * W  (注意：我们还没实现 matmul 的求导，所以这里先用 * 逐元素相乘代替做测试！)
    tensor::Tensor A = ops::ReLU(X);
    tensor::Tensor Y = A * W;

    // 4. 计算 MSE Loss： Loss = mean( (Y - Target) * (Y - Target) )
    tensor::Tensor Diff = Y - Target;
    tensor::Tensor SqDiff = Diff * Diff;
    tensor::Tensor Loss = ops::mean(SqDiff); // 这里调用的就是刚刚写的归约算子

    std::cout << "Forward Pass Complete. MSE Loss: " << Loss.data()[0] << "\n" << std::endl;

    // 5. 终极奥义：反向传播！
    std::cout << "Running Loss.backward()..." << std::endl;
    Loss.backward();

    // 6. 见证奇迹：查看 W 的梯度
    // 如果梯度不是 0，说明反向传播的信号完美穿透了 mean -> 乘法 -> 减法 -> 乘法，到达了 W！
    std::cout << "Gradient of W (dW):\n";
    std::cout << "[" << W.grad()[0] << ", " << W.grad()[1] << "]\n";
    std::cout << "[" << W.grad()[2] << ", " << W.grad()[3] << "]\n";

    return 0;
}