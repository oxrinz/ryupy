#include "../Tensor.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "kernels/Kernels.h"
#include <numeric>

namespace ryupy
{
    std::shared_ptr<Tensor> Tensor::negate()
    {
        std::shared_ptr<Tensor> tensor = handleInPlaceEmptyOperator(negateKernel);

        return tensor;
    }

    std::shared_ptr<Tensor> Tensor::log() const
    {
        std::shared_ptr<Tensor> tensor = handleEmptyOperator(logKernel);
        return tensor;
    }

    std::shared_ptr<Tensor> Tensor::sum(const std::optional<int> &dim, bool keepdim)
    {
        // Full reduction if dim is not specified
        if (!dim.has_value())
        {
            std::vector<int> out_shape = {1};
            auto result = std::make_shared<Tensor>(out_shape);
            result->requires_grad = this->requires_grad;

            // Initialize output to 0
            cudaMemset(result->d_data, 0, sizeof(float));

            // For scalar input, just copy the value
            if (shape.size() == 1 && shape[0] == 1)
            {
                cudaMemcpy(result->d_data, d_data, sizeof(float), cudaMemcpyDeviceToDevice);
            }
            else
            {
                // Setup kernel launch parameters
                int blockSize = 256;
                int numElements = this->size / sizeof(float);
                int numBlocks = (numElements + blockSize - 1) / blockSize;
                int sharedMemSize = blockSize * sizeof(float);

                // Launch kernel with number of elements
                sumKernel<<<numBlocks, blockSize, sharedMemSize>>>(
                    this->d_data, result->d_data, numElements);

                cudaDeviceSynchronize();
            }

            if (requires_grad)
            {
                result->prev = {shared_from_this()};
                result->is_leaf = false;
                result->backward_fn = [this]()
                {
                    if (!this->grad)
                    {
                        this->grad = Tensor::zeros(this->shape, false);
                    }
                    // Gradient of sum is always 1 for all inputs
                    auto ones = Tensor::ones(this->shape, false);
                    *this->grad += *ones;
                };
            }
            return result;
        }

        // Handle negative dimensions
        int reduce_dim = dim.value();
        if (reduce_dim < 0)
            reduce_dim += shape.size();

        if (reduce_dim < 0 || reduce_dim >= shape.size())
            throw std::out_of_range("Invalid reduction dimension");

        // Calculate output shape
        std::vector<int> out_shape;
        if (keepdim)
        {
            out_shape = shape;
            out_shape[reduce_dim] = 1;
        }
        else
        {
            for (int i = 0; i < shape.size(); i++)
            {
                if (i != reduce_dim)
                {
                    out_shape.push_back(shape[i]);
                }
            }
        }

        auto result = std::make_shared<Tensor>(out_shape);
        result->requires_grad = requires_grad;

        // Calculate input strides
        std::vector<int> in_strides = calculate_strides(shape);

        // Allocate device memory and copy data
        int *d_in_strides, *d_out_shape;
        cudaMalloc(&d_in_strides, shape.size() * sizeof(int));
        cudaMalloc(&d_out_shape, out_shape.size() * sizeof(int));
        cudaMemcpy(d_in_strides, in_strides.data(),
                   shape.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_out_shape, out_shape.data(),
                   out_shape.size() * sizeof(int), cudaMemcpyHostToDevice);

        // Launch kernel
        int blockSize = 256;
        int numElements = result->size / sizeof(float);
        int numBlocks = (numElements + blockSize - 1) / blockSize;

        std::vector<int> in_shape = shape;
        int *d_in_shape;
        cudaMalloc(&d_in_shape, shape.size() * sizeof(int));
        cudaMemcpy(d_in_shape, in_shape.data(), shape.size() * sizeof(int), cudaMemcpyHostToDevice);

        sumDimKernel<<<numBlocks, blockSize>>>(
            d_data, result->d_data,
            d_in_strides, d_out_shape,
            d_in_shape,
            reduce_dim, shape[reduce_dim], numElements,
            shape.size());

        cudaFree(d_in_shape);
        cudaDeviceSynchronize();
        cudaFree(d_in_strides);
        cudaFree(d_out_shape);

        if (requires_grad)
        {
            result->prev = {shared_from_this()};
            result->is_leaf = false;
            result->backward_fn = [this]()
            {
                if (!this->grad)
                {
                    this->grad = Tensor::zeros(this->shape, false);
                }
                // Gradient of sum is always 1 for all inputs, regardless of reduction axis
                auto ones = Tensor::ones(this->shape, false);
                *this->grad += *ones;
            };
        }

        return result;
    }

}
