#include "../Tensor.h"
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace ryupy
{
    void Tensor::backward()
    {
        if (!requires_grad)
        {
            throw std::runtime_error("Tensor does not require gradients");
        }

        if (!backward_fn)
        {
            throw std::runtime_error("Tensor has no backward function");
        }

        if (!grad)
        {
            // If this is the output tensor, initialize grad with ones
            grad = std::make_shared<Tensor>(shape);
            // Fill grad with ones
            std::vector<float> ones(size, 1.0f);
            cudaMemcpy(grad->d_data, ones.data(), size * sizeof(float), cudaMemcpyHostToDevice);
        }

        backward_fn();
    }
}