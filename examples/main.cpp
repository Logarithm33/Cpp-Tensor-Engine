#include <iostream>
#include "tensor/tensor.hpp"

int main() {
    std::cout << "Starting cpp-tensor-engine v0.1.0..." << std::endl;
    
    tensor::Tensor t;
    t.print_info();
    
    return 0;
}