#include "ops/math_ops.hpp"
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <vector>
#include <stdexcept>

namespace tensor {
    template <typename FwdFunc, typename BackFuncA, typename BackFuncB>
    Tensor broadcast_apply(const Tensor& A, const Tensor& B, FwdFunc fwd_fn, BackFuncA back_a_fn, BackFuncB back_b_fn) {
        int ndims_a = A.shape().size();
        int ndims_b = B.shape().size();
        int max_dims = std::max(ndims_a, ndims_b);

        std::vector<size_t> pad_shape_a = A.shape();
        pad_shape_a.insert(pad_shape_a.begin(), max_dims - ndims_a, 1);
        std::vector<size_t> pad_strides_a = A.strides();
        pad_strides_a.insert(pad_strides_a.begin(), max_dims - ndims_a, 0);

        std::vector<size_t> pad_shape_b = B.shape();
        pad_shape_b.insert(pad_shape_b.begin(), max_dims - ndims_b, 1);
        std::vector<size_t> pad_strides_b = B.strides();
        pad_strides_b.insert(pad_strides_b.begin(), max_dims - ndims_b, 0);

        std::vector<size_t> out_shape(max_dims);
        for (int i = 0; i < max_dims; ++i) {
            if (pad_shape_a[i] != pad_shape_b[i]) {
                if (pad_shape_a[i] == 1) pad_strides_a[i] = 0;
                else if (pad_shape_b[i] == 1) pad_strides_b[i] = 0;
                else throw std::invalid_argument("Broadcasting error: Incompatible shapes!");
            }
            out_shape[i] = std::max(pad_shape_a[i], pad_shape_b[i]);
        }

        bool requires_grad = A.requires_grad() || B.requires_grad();

        Tensor result(out_shape, requires_grad);
        const float* a_ptr = A.data(); 
        const float* b_ptr = B.data();
        float* res_ptr = result.data();

        for (size_t i = 0; i < result.size(); ++i) {
            size_t temp_idx = i;
            size_t offset_a = 0;
            size_t offset_b = 0;

            for (int d = max_dims - 1; d >= 0; --d) {
                size_t coord = temp_idx % out_shape[d];
                temp_idx /= out_shape[d];
                offset_a += coord * pad_strides_a[d];
                offset_b += coord * pad_strides_b[d];
            }

            res_ptr[i] = fwd_fn(a_ptr[offset_a], b_ptr[offset_b]);
        }

        if (!requires_grad) {
            return result;
        }

        std::vector<Tensor> prev;
        if (A.requires_grad()) prev.push_back(A);
        if (B.requires_grad()) prev.push_back(B);
        result.set_prev(prev);

        auto backward_fn = [node_a = A, node_b = B, result, 
                    out_shape, pad_strides_a, pad_strides_b, 
                    back_a_fn, back_b_fn]() mutable {
    
            float *grad_a = node_a.requires_grad() ? node_a.grad() : nullptr;
            float *grad_b = node_b.requires_grad() ? node_b.grad() : nullptr;
            const float* grad_out = result.grad();
            const float* data_a = node_a.data();
            const float* data_b = node_b.data();

            int max_dims = out_shape.size();

            for (size_t i = 0; i < result.size(); ++i) {
                size_t temp_idx = i;
                size_t offset_a = 0;
                size_t offset_b = 0;

                for (int d = max_dims - 1; d >= 0; --d) {
                    size_t coord = temp_idx % out_shape[d];
                    temp_idx /= out_shape[d];
                    offset_a += coord * pad_strides_a[d];
                    offset_b += coord * pad_strides_b[d];
                }

                if (grad_a) {
                    grad_a[offset_a] += grad_out[i] * back_a_fn(data_a[offset_a], data_b[offset_b]);
                }
                if (grad_b) {
                    grad_b[offset_b] += grad_out[i] * back_b_fn(data_a[offset_a], data_b[offset_b]);
                }
            }
        };

        result.set_backward_fn(backward_fn);
        return result;
    }

    Tensor operator+(const Tensor& a, const Tensor& b) {
        return broadcast_apply(a, b, 
            [](float x, float y) { return x + y; },
            [](float x, float y) { return 1.0f; }, 
            [](float x, float y) { return 1.0f; }
        );
    }

    Tensor operator-(const Tensor& a, const Tensor& b) {
        return broadcast_apply(a, b, 
            [](float x, float y) { return x - y; },
            [](float x, float y) { return 1.0f; }, 
            [](float x, float y) { return -1.0f; }
        );
    }

    Tensor operator*(const Tensor& a, const Tensor& b) {
        return broadcast_apply(a, b, 
            [](float x, float y) { return x * y; },
            [](float x, float y) { return y; }, 
            [](float x, float y) { return x; }
        );
    }

    Tensor operator/(const Tensor& a, const Tensor& b) {
        return broadcast_apply(a, b, 
            [](float x, float y) { return x / y; },
            [](float x, float y) { return 1.0f / y; }, 
            [](float x, float y) { return -x / (y * y); }
        );
    }
}

namespace ops {
    tensor::Tensor sum(const tensor::Tensor& t) {
        float sum_val = 0.0f;
        if (t.data() && t.size() > 0) {
            for (size_t i = 0; i < t.size(); ++i) {
                sum_val = std::accumulate(t.data(), t.data() + t.size(), 0.0f);
            }
        }

        tensor::Tensor result({1}, t.requires_grad());
        result.data()[0] = sum_val;
        
        if (!t.requires_grad()) {
            return result;
        }

        result.set_prev({t});

        auto backward_fn = [in_node = t, result]() mutable {
            float* grad_in = in_node.grad();
            const float* grad_out = result.grad();

            if (grad_in) {
                for (size_t i = 0; i < in_node.size(); ++i) {
                    grad_in[i] += grad_out[0] * 1.0f;
                }
            }
        };
        result.set_backward_fn(backward_fn);
        return result;
    }

    tensor::Tensor mean(const tensor::Tensor& t) {
        if (t.size() == 0) {
            throw std::runtime_error("Mean error: Cannot calculate mean of an empty tensor.");
        }
        
        float sum_val = std::accumulate(t.data(), t.data() + t.size(), 0.0f);
        float mean_val = sum_val / static_cast<float>(t.size());

        tensor::Tensor result({1}, t.requires_grad());
        result.data()[0] = mean_val;
        
        if (!t.requires_grad()) {
            return result;
        }

        result.set_prev({t});

        auto backward_fn = [in_node = t, result]() mutable {
            float* grad_in = in_node.grad();
            const float* grad_out = result.grad();

            if (grad_in) {
                for (size_t i = 0; i < in_node.size(); ++i) {
                    grad_in[i] += grad_out[0] / static_cast<float>(in_node.size());
                }
            }
        };
        result.set_backward_fn(backward_fn);
        return result;
    }

    tensor::Tensor matmul(const tensor::Tensor& a, const tensor::Tensor& b) {
        if (a.shape().size() != 2 || b.shape().size() != 2) {
            throw std::invalid_argument("Matmul error: both tensors must be 2D.");
        }
        if (a.shape()[1] != b.shape()[0]) {
            throw std::invalid_argument("Matmul error: inner dimensions must match.");
        }

        size_t a_row = a.shape()[0];
        size_t a_col = a.shape()[1];
        size_t b_col = b.shape()[1];

        bool requires_grad = a.requires_grad() || b.requires_grad();
        tensor::Tensor result({a_row, b_col}, requires_grad);
        result.fill(0.0f);
        
        const float* a_ptr = a.data();
        const float* b_ptr = b.data();
        float* c_ptr = result.data();
        
        const auto& a_strides = a.strides();
        const auto& b_strides = b.strides();
        const auto& res_strides = result.strides();
        
        for (size_t i = 0; i < a_row; ++i) {
            for (size_t p = 0; p < a_col; ++p) {
                float a_val = a_ptr[i * a_strides[0] + p * a_strides[1]];
                
                for (size_t j = 0; j < b_col; ++j) {
                    c_ptr[i * res_strides[0] + j * res_strides[1]] += 
                        a_val * b_ptr[p * b_strides[0] + j * b_strides[1]];
                }
            }
        }
        
        if (!requires_grad) {
            return result;
        }

        std::vector<tensor::Tensor> prev;
        if (a.requires_grad()) prev.push_back(a);
        if (b.requires_grad()) prev.push_back(b);
        result.set_prev(prev);

        auto backward_fn = [node_a = a, node_b = b, result,
                            a_row, a_col, b_col, a_strides, b_strides, res_strides]() mutable {

            float* grad_a = node_a.requires_grad() ? node_a.grad() : nullptr;
            float* grad_b = node_b.requires_grad() ? node_b.grad() : nullptr;
            const float* grad_out = result.grad();
            const float* data_a = node_a.data();
            const float* data_b = node_b.data();

            for (size_t i = 0; i < a_row; ++i) {
                for (size_t j = 0; j < b_col; ++j) {
                    float grad_out_val = grad_out[i * res_strides[0] + j * res_strides[1]];
                    for (size_t k = 0; k < a_col; ++k) {
                        if (grad_a) {
                            grad_a[i * a_strides[0] + k * a_strides[1]] += 
                                grad_out_val * data_b[k * b_strides[0] + j * b_strides[1]];
                        }
                        if (grad_b) {
                            grad_b[k * b_strides[0] + j * b_strides[1]] += 
                                grad_out_val * data_a[i * a_strides[0] + k * a_strides[1]];
                        }
                    }
                }
            }
        };

        result.set_backward_fn(backward_fn);
        return result;
    }

    tensor::Tensor CrossEntropyWithLogits(const tensor::Tensor& logits, const tensor::Tensor& targets) {
        if (logits.shape().size() != 2 || targets.shape().size() != 2) {
            throw std::invalid_argument("CrossEntropy error: logits and targets must be 2D [Batch, Classes].");
        }
        
        size_t batch_size = logits.shape()[0];
        size_t num_classes = logits.shape()[1];

        const float* logits_ptr = logits.data();
        const float* targets_ptr = targets.data();
        
        const auto& log_strides = logits.strides();
        const auto& tgt_strides = targets.strides();

        // 临时数组：保存计算出来的 Softmax 概率，这是反向传播唯一需要的东西！
        // (在闭包捕获时，这个 vector 会被自动拷贝进闭包，充当 Save for backward)
        std::vector<float> probs(batch_size * num_classes, 0.0f);
        float total_loss = 0.0f;

        // 1. 前向传播：安全计算 Softmax 和 Loss
        for (size_t i = 0; i < batch_size; ++i) {
            
            // 步骤 A：寻找当前样本的最大 Logit (Max-Trick 防溢出)
            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < num_classes; ++j) {
                float val = logits_ptr[i * log_strides[0] + j * log_strides[1]];
                if (val > max_val) max_val = val;
            }

            // 步骤 B：计算指数和 (Sum of Exp)
            float sum_exp = 0.0f;
            for (size_t j = 0; j < num_classes; ++j) {
                // 核心：减去 max_val
                float e = std::exp(logits_ptr[i * log_strides[0] + j * log_strides[1]] - max_val);
                probs[i * num_classes + j] = e; 
                sum_exp += e;
            }

            // 步骤 C：归一化得到最终概率，并计算交叉熵 Loss
            for (size_t j = 0; j < num_classes; ++j) {
                probs[i * num_classes + j] /= sum_exp; // P = e^z / sum
                
                float target_val = targets_ptr[i * tgt_strides[0] + j * tgt_strides[1]];
                
                // Cross Entropy 公式： - sum( Y * log(P) )
                if (target_val > 0.0f) {
                    // + 1e-7f 也是工业界惯例，防止由于极度接近0导致的 log(0) 崩溃
                    total_loss -= target_val * std::log(probs[i * num_classes + j] + 1e-7f);
                }
            }
        }

        // 取 Batch 的平均 Loss
        total_loss /= static_cast<float>(batch_size);

        // 2. 创建标量输出张量
        tensor::Tensor result({1}, logits.requires_grad());
        result.data()[0] = total_loss;

        if (!logits.requires_grad()) return result;

        // 3. 构建计算图记忆
        result.set_prev({logits});

        // 4. 见证奇迹：反向传播闭包
        // 我们甚至不需要捕获 targets_ptr，只需把 targets 对象按值捕获进来
        auto backward_fn = [in_node = logits, out_node = result, 
                            probs, targets, batch_size, num_classes, log_strides, tgt_strides]() mutable {
            
            float* grad_in = in_node.grad();
            const float* grad_out = out_node.grad();
            const float* tgt_ptr = targets.data();

            if (grad_in) {
                float g_out = grad_out[0]; // 接收最顶层的梯度 (通常是 1.0)
                
                for (size_t i = 0; i < batch_size; ++i) {
                    for (size_t j = 0; j < num_classes; ++j) {
                        size_t p_idx = i * num_classes + j;
                        size_t tgt_idx = i * tgt_strides[0] + j * tgt_strides[1];
                        size_t grad_idx = i * log_strides[0] + j * log_strides[1];

                        // 极致暴力的数学之美：导数 dZ = (P - Y) / N
                        float local_grad = (probs[p_idx] - tgt_ptr[tgt_idx]) / static_cast<float>(batch_size);
                        
                        // 链式法则累加
                        grad_in[grad_idx] += g_out * local_grad;
                    }
                }
            }
        };

        result.set_backward_fn(backward_fn);
        return result;
    }
}