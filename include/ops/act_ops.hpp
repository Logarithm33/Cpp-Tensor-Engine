#pragma once
#include "tensor/tensor.hpp"

namespace ops {
    tensor::Tensor ReLU(const tensor::Tensor& t);

    tensor::Tensor Sigmoid(const tensor::Tensor& t);
}