#include "ops/act_ops.hpp"
#include <algorithm>
#include <cmath>

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
}