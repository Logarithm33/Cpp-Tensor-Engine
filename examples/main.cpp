#include <iostream>
#include "tensor/tensor.hpp"
#include "ops/math_ops.hpp" // 引入需要的算子库
#include "ops/act_ops.hpp"

int main() {
    std::cout << "--- 工业级解耦架构测试 ---" << std::endl;
    
    // 容器只负责数据生成
    tensor::Tensor X = tensor::Tensor::randn({2, 3});
    tensor::Tensor W = tensor::Tensor::randn({3, 4});
    tensor::Tensor b = tensor::Tensor::randn({4});

    // 动作全部由 ops 库接管！
    // 以前是 X.matmul(W)，现在变成了符合数学直觉的 ops::matmul(X, W)
    tensor::Tensor Z = ops::matmul(X, W) + b;
    
    // 以前是 Z.relu()，现在是 ops::relu(Z)
    tensor::Tensor A = ops::ReLU(Z);

    std::cout << "Activated Output:\n" << A << std::endl;

    return 0;
}