#include "ops/act_ops.hpp"
#include <algorithm>

namespace ops {
    tensor::Tensor ReLU(const tensor::Tensor& t) {
        tensor::Tensor result(t.shape());

        const float* src_ptr = t.data();
        float* dst_ptr = result.data();

        for (size_t i = 0; i < t.size(); ++i) {
            dst_ptr[i] = std::max(0.0f, src_ptr[i]);
        }
        return result;
    }
}