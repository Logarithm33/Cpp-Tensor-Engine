#include "ops/math_ops.hpp"
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace tensor {
    template <typename Func>
    Tensor broadcast_apply(const Tensor& A, const Tensor& B, Func op) {
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

        Tensor result(out_shape);
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

            res_ptr[i] = op(a_ptr[offset_a], b_ptr[offset_b]);
        }
        return result;
    }

    Tensor operator+(const Tensor& a, const Tensor& b) {
        return broadcast_apply(a, b, [](float x, float y) { return x + y; });
    }

    Tensor operator-(const Tensor& a, const Tensor& b) {
        return broadcast_apply(a, b, [](float x, float y) { return x - y; });
    }

    Tensor operator*(const Tensor& a, const Tensor& b) {
        return broadcast_apply(a, b, [](float x, float y) { return x * y; });
    }

    Tensor operator/(const Tensor& a, const Tensor& b) {
        return broadcast_apply(a, b, [](float x, float y) { return x / y; });
    }
}

namespace ops {
    float sum(const tensor::Tensor& t) {
        if (!t.data() || t.size() == 0) return 0.0f;
        const float* ptr = t.data();
        return std::accumulate(ptr, ptr + t.size(), 0.0f);
    }

    float mean(const tensor::Tensor& t) {
        if (t.size() == 0) {
            throw std::runtime_error("Mean error: Cannot calculate mean of an empty tensor.");
        }
        return sum(t) / static_cast<float>(t.size());
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

        tensor::Tensor result({a_row, b_col});
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
        return result;
    }
}