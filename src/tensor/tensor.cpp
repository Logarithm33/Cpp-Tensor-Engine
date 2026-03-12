#include "tensor/tensor.hpp"
#include <iostream>
#include <utility>

namespace tensor {
    Tensor::Tensor() : shape_({0}), strides_({0}), size_(0), data_(nullptr) {}

    Tensor::Tensor(std::vector<size_t> shape)
        : shape_(std::move(shape)) {

            size_ = 1;
            for (size_t dim: shape_) {
                size_ *= dim;
            }

            if (size_ > 0) {
                data_ = std::shared_ptr<float[]>(new float[size_]);
            }
            else {
                data_ = nullptr;
            }
            compute_strides();
        }

        Tensor::Tensor(std::vector<size_t> shape, std::vector<size_t> strides, std::shared_ptr<float[]> data)
            : shape_(std::move(shape)), strides_(std::move(strides)),data_(std::move(data)) {
            size_ = 1;
            for (size_t dim: shape_) {
                size_ *=dim;
            }
        }

        void Tensor::compute_strides() {
            strides_.resize(shape_.size());

            if (shape_.empty()) return;

            strides_.back() = 1;

            for (int i = shape_.size() - 2; i >=0; --i) {
                strides_[i] = strides_[i+1] *shape_[i+1];
            }
        }

    void Tensor::print_info() const {
        std::cout << "Tensor Shape: [";
        for (size_t i = 0; i < shape_.size(); ++i) {
            std::cout << shape_[i] << (i == shape_.size() - 1 ? "" : ", ");
        }
        std::cout << "]\nStrides: [";
        for (size_t i = 0; i < strides_.size(); ++i) {
            std::cout << strides_[i] << (i == strides_.size() - 1 ? "" : ", ");
        }
        std::cout << "]\nTotal Size: " << size_ << "\n" << std::endl;
    }
}