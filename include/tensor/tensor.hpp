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

        bool requires_grad_;
        std::shared_ptr<float[]> grad_;

        void compute_strides();

    public:
        Tensor();
        Tensor(std::vector<size_t> shape, bool requires_grad = false);
        Tensor(std::vector<size_t> shape, std::vector<size_t> strides, std::shared_ptr<float[]> data, bool requires_grad = false);
        static Tensor randn(std::vector<size_t> shape ,float mean = 0.0f, float stddev = 1.0f, bool requires_grad = false);

        void fill(float value);
        Tensor reshape(std::vector<size_t> new_shape) const;

        Tensor Transpose() const;
        
        const std::vector<size_t>& shape() const { return shape_; }
        const std::vector<size_t>& strides() const { return strides_; }
        size_t size() const { return size_; }  
        float* data() { return data_.get(); }
        const float* data() const { return data_.get(); }
        
        bool requires_grad() const { return requires_grad_; }
        float* grad() { return grad_.get(); }
        const float* grad() const { return grad_.get(); }

        void zero_grad();
    };

    std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
}