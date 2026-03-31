#pragma once
#include "tensor/tensor.hpp"

namespace nn {
    class Linear {
    private:
        size_t in_features_;
        size_t out_features_;

        tensor::Tensor weight_;
        tensor::Tensor bias_;
    public: 
        Linear(size_t in_features, size_t out_features);
        tensor::Tensor forward(const tensor::Tensor& input) const;

        tensor::Tensor& weight() { return weight_; }
        tensor::Tensor& bias() {return bias_; }
    };
}
