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
    
    
        void Tensor::fill(float value) {
        if (!data_) return;

        for (size_t i = 0; i < size_; ++i) {
            data_[i] = value;
        }
    }

    Tensor Tensor::reshape(std::vector<size_t> new_shape) const {
        size_t new_size = 1;
        for (size_t dim: new_shape) {
            new_size *= dim;
        }

        if (new_size != size_) {
            throw std::invalid_argument("Reshape error: total size must remain the same.");
        }

        std::vector<size_t> new_strides(new_shape.size());
        if (!new_shape.empty()) {
            new_strides.back() = 1;
            for (int i = new_shape.size() - 2; i >= 0; --i) {
                new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
            }
        }

        return Tensor(new_shape, new_strides, data_);
    }

    Tensor Tensor::operator+(const Tensor& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Addition error: Tensors must have the same shape.");
        }

        Tensor result(shape_);

        float* result_ptr = result.data_.get();
        float* a_ptr = this->data_.get();
        float* b_ptr = other.data_.get();


        for (size_t i = 0; i < size_; ++i) {
            result_ptr[i] = a_ptr[i] + b_ptr[i];
        }
        return result;
    }
    void Tensor::print_data() const {
        if (!data_) {
            std::cout << "Tensor is empty." << std::endl;
            return;
        }

        std::cout << "Tensor Data: [";
        for (size_t i = 0; i < size_; ++i) {
            std::cout << data_[i] << (i == size_ - 1 ? "" : ", ");
        }
        std::cout << "]\n" << std::endl;
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