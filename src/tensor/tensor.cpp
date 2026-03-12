#include "tensor/tensor.hpp"
#include <iostream>

namespace tensor {
    Tensor::Tensor() {
    }

    void Tensor::print_info() const {
        std::cout << "Tensor Engine Initialized Successfully!" << std::endl;
    }
}