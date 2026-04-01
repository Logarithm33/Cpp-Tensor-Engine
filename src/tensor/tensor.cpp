#include "tensor/tensor.hpp"
#include <iostream>
#include <utility>
#include <random>
#include <iomanip>
#include <algorithm>
#include <numeric>

namespace tensor {
    void Tensor::compute_strides() {
            strides_.resize(shape_.size());

            if (shape_.empty()) return;

            strides_.back() = 1;

            for (int i = shape_.size() - 2; i >=0; --i) {
                strides_[i] = strides_[i+1] *shape_[i+1];
            }
        }

    Tensor::Tensor() : shape_({0}), strides_({0}), size_(0), data_(nullptr), requires_grad_(false), grad_(nullptr) {}

    Tensor::Tensor(std::vector<size_t> shape, bool requires_grad)
        : shape_(std::move(shape)), requires_grad_(requires_grad) {

            size_ = 1;
            for (size_t dim: shape_) {
                size_ *= dim;
            }

            if (size_ > 0) {
                data_ = std::shared_ptr<float[]>(new float[size_]);

                if (requires_grad_) {
                    grad_ = std::shared_ptr<float[]>(new float[size_]);
                    zero_grad();
                } else {
                    grad_ = nullptr;
                }
            } else {
                data_ = nullptr;
                grad_ = nullptr;
            }
            compute_strides();
        }
   
    Tensor::Tensor(std::vector<size_t> shape, std::vector<size_t> strides, std::shared_ptr<float[]> data, bool requires_grad)
            : shape_(std::move(shape)), strides_(std::move(strides)),data_(std::move(data)), requires_grad_(requires_grad), grad_(nullptr) {
            size_ = 1;
            for (size_t dim: shape_) {
                size_ *=dim;
            }
        }

    void Tensor::zero_grad() {
        if (size_ > 0 && grad_) {
            std::fill(grad_.get(), grad_.get() + size_, 0.0f);
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

    Tensor Tensor::Transpose() const {
        if (shape_.size() != 2) {
            throw std::invalid_argument("Transpose error: only 2D tensors are supported.");
        }

        std::vector<size_t> new_shape = {this->shape_[1], this->shape_[0]};
        std::vector<size_t> new_strides = {this->strides_[1], this->strides_[0]};

        return Tensor(new_shape, new_strides, this->data_);
    }

    Tensor Tensor::randn(std::vector<size_t> shape, float mean, float stddev, bool requires_grad) {
        Tensor result(shape, requires_grad);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(mean, stddev);
        
        float* ptr = result.data();
        for (size_t i = 0; i < result.size_; ++i) {
            ptr[i] = dist(gen);
        }
        return result;
    }

    std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        const auto& shape = tensor.shape();
        const auto& strides = tensor.strides();
        const float* ptr = tensor.data();

        if (shape.empty() || ptr == nullptr) {
            return os << "Empty Tensor";
        }

        os << std::fixed << std::setprecision(4);

        if (shape.size() == 1) {
            os << "[";
            for (size_t i = 0; i < shape[0]; ++i) {
                os << std::setw(8) << ptr[i * strides[0]] << (i == shape[0] - 1 ? "" : ", ");
            }
            os << "]";
        }
        else if (shape.size() == 2) {
            os << "[\n";
            for (size_t i = 0; i < shape[0]; ++i) {
                os << "  [";
                for (size_t j = 0; j < shape[1]; ++j) {
                    os << std::setw(8) << ptr[i * strides[0] + j * strides[1]] << (j == shape[1] - 1 ? "" : ", ");
                }
                os << "]\n";
            }
            os << "]";
        }
        else {
            os << "Tensor(rank=" << shape.size() << ", size=" << tensor.size() << ")";
        }
        return os;
        
    }
}