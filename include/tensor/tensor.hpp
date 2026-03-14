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

        const std::vector<size_t>& shape() const { return shape_; }
        const std::vector<size_t>& strides() const { return strides_; }
        size_t size() const { return size_; }
        
        float* data() { return data_.get(); }

        const float* data() const { return data_.get(); }

        void fill(float value);
        Tensor reshape(std::vector<size_t> new_shape) const;

        Tensor matmul(const Tensor& other) const;

        Tensor operator+(const Tensor& other) const;
        void print_data() const;
        void print_info() const;
    };
}