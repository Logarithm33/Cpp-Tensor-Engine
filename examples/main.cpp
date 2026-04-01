#include <iostream>
#include "tensor/tensor.hpp"
#include "ops/math_ops.hpp"

int main() {
    std::cout << "--- 动态计算图自动求导测试 (Autograd) ---" << std::endl;
    
    // A 需要求导 (requires_grad = true)
    tensor::Tensor A({2}, true); 
    A.data()[0] = 2.0f; A.data()[1] = 3.0f;
    
    // B 和 C 是普通常量，不需要求导
    tensor::Tensor B({2}); 
    B.data()[0] = 5.0f; B.data()[1] = 4.0f;
    
    tensor::Tensor C({2}); 
    C.data()[0] = 10.0f; C.data()[1] = 10.0f;

    // 1. 前向传播：构建计算图  L = A * B + C
    // L = [2*5 + 10, 3*4 + 10] = [20, 22]
    tensor::Tensor M = A * B;
    tensor::Tensor L = M + C;

    std::cout << "Forward Pass Output L:\n" << L << "\n" << std::endl;

    // 2. 魔法时刻：一键反向传播！
    L.backward();

    // 3. 验证梯度：dL / dA
    // L = A * B + C 
    // 根据微积分：dL / dA = B
    // 所以 A 的梯度应该完美等于 B 的数据：[5.0, 4.0]
    std::cout << "Gradients of A (should be equal to B):\n";
    std::cout << "[" << A.grad()[0] << ", " << A.grad()[1] << "]" << std::endl;

    return 0;
}