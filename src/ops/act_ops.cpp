#include "ops/act_ops.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ops {
    tensor::Tensor ReLU(const tensor::Tensor& t) {
        tensor::Tensor result(t.shape(), t.requires_grad());

        const float* src_ptr = t.data();
        float* dst_ptr = result.data();

        for (size_t i = 0; i < t.size(); ++i) {
            dst_ptr[i] = std::max(0.0f, src_ptr[i]);
        }
        
        if (!t.requires_grad()) {
            return result;
        }

        result.set_prev({t});

        auto backward_fn = [in_node = t, out_node = result]() mutable {
            float* grad_in = in_node.grad();
            const float* grad_out = out_node.grad();
            const float* data_in = in_node.data();

            for (size_t i = 0; i < in_node.size(); ++i) {
                grad_in[i] += (data_in[i] > 0.0f) ? grad_out[i] : 0.0f;
            }
        };

        result.set_backward_fn(backward_fn);
        return result;
    }

    tensor::Tensor Sigmoid(const tensor::Tensor& t) {
        tensor::Tensor result(t.shape(), t.requires_grad());

        const float* src_ptr = t.data();
        float* dst_ptr = result.data();

        for (size_t i = 0; i < t.size(); ++i) {
            dst_ptr[i] = 1.0f / (1.0f + std::exp(-src_ptr[i]));
        }
        
        if (!t.requires_grad()) {
            return result;
        }

        result.set_prev({t});

        auto backward_fn = [in_node = t, out_node = result]() mutable {
            float* t_grad_ptr = in_node.grad();
            const float* grad_out = out_node.grad();
            const float* data_out = out_node.data();

            if (t_grad_ptr) {
                for (size_t i = 0; i < out_node.size(); ++i) {
                    t_grad_ptr[i] += grad_out[i] * data_out[i] * (1.0f - data_out[i]);
                }
            }
        };

        result.set_backward_fn(backward_fn);
        return result;
    }

    tensor::Tensor Softmax(const tensor::Tensor& t) {
        if (t.requires_grad()) {
            throw std::runtime_error(
                "Architecture Warning: Standalone Softmax does not support backward. "
                "Please use fused 'CrossEntropyWithLogits' for training!"
            );
        }

        if (t.shape().size() != 2) {
            throw std::invalid_argument("Softmax currently only supports 2D batched input.");
        }

        size_t batch_size = t.shape()[0];
        size_t num_classes = t.shape()[1];

        tensor::Tensor result(t.shape(), false); // 强制不需要求导
        
        const float* in_ptr = t.data();
        float* out_ptr = result.data();
        const auto& strides = t.strides();

        for (size_t i = 0; i < batch_size; ++i) {
            // 1. 找最大值 (Max-Trick)
            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < num_classes; ++j) {
                float val = in_ptr[i * strides[0] + j * strides[1]];
                if (val > max_val) max_val = val;
            }

            // 2. 算指数和
            float sum_exp = 0.0f;
            for (size_t j = 0; j < num_classes; ++j) {
                float e = std::exp(in_ptr[i * strides[0] + j * strides[1]] - max_val);
                out_ptr[i * strides[0] + j * strides[1]] = e;
                sum_exp += e;
            }

            // 3. 归一化为概率
            for (size_t j = 0; j < num_classes; ++j) {
                out_ptr[i * strides[0] + j * strides[1]] /= sum_exp;
            }
        }

        return result;
    }
}