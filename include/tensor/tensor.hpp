#pragma once

#include <vector>
#include <memory>

namespace tensor {

    class Tensor {
    private:
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;
        size_t size_;
        std::shared_ptr<float[]> data_;

        void compute_strides();

    public:
        Tensor();
        Tensor(std::vector<size_t> shape);
        Tensor(std::vector<size_t> shape, std::vector<size_t> strides, std::shared_ptr<float[]> data);
        void print_info() const;
    };
}