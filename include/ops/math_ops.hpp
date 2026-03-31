#pragma once
#include "tensor/tensor.hpp"

namespace tensor {
    Tensor operator+(const Tensor& a, const Tensor& b);
    Tensor operator-(const Tensor& a, const Tensor& b);
    Tensor operator*(const Tensor& a, const Tensor& b);
    Tensor operator/(const Tensor& a, const Tensor& b);
}

namespace ops {
    tensor::Tensor matmul(const tensor::Tensor& a, const tensor::Tensor& b);
    float sum(const tensor::Tensor& t);
    float mean(const tensor::Tensor& t);
}