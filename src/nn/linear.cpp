#include "tensor/tensor.hpp"
#include "nn/linear.hpp"
#include "ops/math_ops.hpp"
#include <cmath>
#include <iostream>

namespace nn {
    Linear::Linear(size_t in_features, size_t out_features)
        : in_features_(in_features), out_features_(out_features) {
        float stddev = std::sqrt(2.0f / in_features_);

        weight_ = tensor::Tensor::randn({in_features_, out_features_}, 0.0f, stddev);
        bias_ = tensor::Tensor({out_features_});
        bias_.fill(0.0f);
    }

    tensor::Tensor Linear::forward(const tensor::Tensor& input) const {
        if (input.shape().size() != 2) {
            throw std::invalid_argument("Linear layer currently only supports 2D batched input [Batch, In_Features]");
        }
        if (input.shape()[1] != in_features_) {
            throw std::invalid_argument("Input feature dimension does not match layer's in_features.");
        }

        return ops::matmul(input, weight_) + bias_;
    }
}


