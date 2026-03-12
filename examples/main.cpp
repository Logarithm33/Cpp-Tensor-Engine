#include <iostream>
#include "tensor/tensor.hpp"

int main() {
    // 创建一个 3维张量，形状为 [2, 3, 4]
    std::vector<size_t> my_shape = {2, 3, 4};
    tensor::Tensor t(my_shape);
    
    t.print_info(); 
    
    return 0;
}