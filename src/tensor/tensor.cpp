#include "tensor/tensor.hpp"
#include <iostream>
#include <utility>
#include <random>
#include <iomanip>

namespace tensor {
    void Tensor::compute_strides() {
            strides_.resize(shape_.size());

            if (shape_.empty()) return;

            strides_.back() = 1;

            for (int i = shape_.size() - 2; i >=0; --i) {
                strides_[i] = strides_[i+1] *shape_[i+1];
            }
        }

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

    Tensor Tensor::matmul(const Tensor& other) const {
        if (shape_.size() != 2 || other.shape_.size() != 2) {
            throw std::invalid_argument("Matmul error: both tensors must be 2D.");
        }
        if (shape_[1] != other.shape_[0]) {
            throw std::invalid_argument("Matmul error: inner dimensions must match.");
        }

        size_t a_row = shape_[0];
        size_t a_col = shape_[1];
        size_t b_col = other.shape_[1];

        Tensor result({a_row, b_col});
        result.fill(0.0f);
        const float* a_ptr = this->data();
        const float* b_ptr = other.data();
        float* c_ptr = result.data();
        
        for (size_t i = 0; i < a_row; ++i) {
            for (size_t p = 0; p < a_col; ++p) {
                float a_val = a_ptr[i * this->strides_[0] + p * this->strides_[1]];
                for (size_t j = 0; j < b_col; ++j) {
                    c_ptr[i * result.strides_[0] + j * result.strides_[1]] += a_val * b_ptr[p * other.strides_[0] + j * other.strides_[1]];
                }
            }
        }
        return result;
    }
    Tensor Tensor::operator+(const Tensor& other) const {
        int a_dims = this->shape_.size();
        int b_dims = other.shape_.size();
        int max_dims = std::max(a_dims, b_dims);

        std::vector<size_t> pad_shape_a = this->shape_;
        pad_shape_a.insert(pad_shape_a.begin(), max_dims - a_dims, 1);
        std::vector<size_t> pad_strides_a = this->strides_;
        pad_strides_a.insert(pad_strides_a.begin(), max_dims - a_dims, 0);

        std::vector<size_t> pad_shape_b = other.shape_;
        pad_shape_b.insert(pad_shape_b.begin(), max_dims - b_dims, 1);
        std::vector<size_t> pad_strides_b = other.strides_;
        pad_strides_b.insert(pad_strides_b.begin(), max_dims - b_dims, 0);

        std::vector<size_t> result_shape(max_dims);
        for (int i = 0; i < max_dims; ++i) {
            if (pad_shape_a[i] != pad_shape_b[i]) {
                if (pad_shape_a[i] == 1) {
                    pad_strides_a[i] = 0;
                }
                else if (pad_shape_b[i] == 1) {
                    result_shape[i] = pad_shape_a[i];
                    pad_strides_b[i] = 0;
                }
                else {
                    throw std::invalid_argument("Shapes cannot be broadcasted for addition.");
                }
            }
            
            result_shape[i] = std::max(pad_shape_a[i], pad_shape_b[i]);
        }
        Tensor result(result_shape);

        for (size_t i = 0; i < result.size_; ++i) {
            size_t idx_a = 0;
            size_t idx_b = 0;
            size_t tmp_idx = i;

            for (int d = max_dims - 1; d >= 0; --d) {
                size_t pos = tmp_idx % result_shape[d];
                tmp_idx /= result_shape[d];
                idx_a += pos * pad_strides_a[d];
                idx_b += pos * pad_strides_b[d];
            }
            result.data_[i] = this->data_[idx_a] + other.data_[idx_b];
        }
        return result;
    }

    Tensor Tensor::ReLU() const {
        Tensor result(this->shape_);

        const float* src_ptr = this->data();
        float* dst_ptr = result.data();
        for (size_t i = 0; i < size_; ++i) {
            dst_ptr[i] = std::max(0.0f, src_ptr[i]);
        }
        return result;
    }

    Tensor Tensor::Transpose() const {
        if (shape_.size() != 2) {
            throw std::invalid_argument("Transpose error: only 2D tensors are supported.");
        }

        std::vector<size_t> new_shape = {this->shape_[1], this->shape_[0]};
        std::vector<size_t> new_strides = {this->strides_[1], this->strides_[0]};

        return Tensor(new_shape, new_strides, this->data_);
    }

    Tensor Tensor::randn(std::vector<size_t> shape, float mean, float stddev) {
        Tensor result(shape);
        
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