#include "../Tensor.h"
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>

namespace ryupy
{
    std::shared_ptr<Tensor> Tensor::handleOperator(Tensor &other, KernelFunc kernel)
    {
        if (shape != other.shape)
        {
            throw std::invalid_argument("Tensors must be same shape");
        }

        auto result = std::make_shared<Tensor>(*this);

        std::cout << "sex " << requires_grad << std::endl;

        if (requires_grad || other.requires_grad)
        {
            result->requires_grad = true;
            result->is_leaf = false;
            result->prev.push_back(shared_from_this());
            result->prev.push_back(other.shared_from_this());
        }

        cudaMalloc(&result->d_data, size);

        cudaMemcpy(result->d_data, this->d_data, size, cudaMemcpyDeviceToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (size / sizeof(float) + threadsPerBlock - 1) / threadsPerBlock;

        kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, other.d_data, result->d_data, size / sizeof(float));

        cudaDeviceSynchronize();

        return result;
    }

    std::shared_ptr<Tensor> Tensor::handleInPlaceOperator(const Tensor &other, KernelFunc kernel)
    {
        if (shape != other.shape)
        {
            throw std::invalid_argument("Tensors must be the same shape");
        }

        int threadsPerBlock = 256;
        int blocksPerGrid = (size / sizeof(float) + threadsPerBlock - 1) / threadsPerBlock;

        kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, other.d_data, d_data, size / sizeof(float));

        cudaDeviceSynchronize();

        return shared_from_this();
    }

    std::shared_ptr<Tensor> Tensor::handleShiftOperator(const int shift, KernelShiftFunc kernel) const
    {
        auto result = std::make_shared<Tensor>(*this);

        cudaMalloc(&result->d_data, size);

        cudaMemcpy(result->d_data, this->d_data, size, cudaMemcpyDeviceToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (size / sizeof(float) + threadsPerBlock - 1) / threadsPerBlock;

        kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, result->d_data, size / sizeof(float), shift);
        cudaDeviceSynchronize();

        return result;
    }

    std::shared_ptr<Tensor> Tensor::handleEmptyOperator(KernelEmptyFunc kernel) const
    {
        auto result = std::make_shared<Tensor>(*this);

        cudaMalloc(&result->d_data, size);

        cudaMemcpy(result->d_data, this->d_data, size, cudaMemcpyDeviceToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (size / sizeof(float) + threadsPerBlock - 1) / threadsPerBlock;

        kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, result->d_data, size / sizeof(float));
        cudaDeviceSynchronize();

        return result;
    }

    std::shared_ptr<Tensor> Tensor::handleInPlaceShiftOperator(int shift, KernelShiftFunc kernel)
    {
        int threadsPerBlock = 256;
        int blocksPerGrid = (size / sizeof(float) + threadsPerBlock - 1) / threadsPerBlock;

        // Apply the kernel directly to this tensor's data
        kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_data, size / sizeof(float), shift);
        cudaDeviceSynchronize();

        return shared_from_this();
    }

    std::shared_ptr<Tensor> Tensor::handleInPlaceEmptyOperator(KernelEmptyFunc kernel)
    {
        int threadsPerBlock = 256;
        int blocksPerGrid = (size / sizeof(float) + threadsPerBlock - 1) / threadsPerBlock;

        // Apply the kernel directly to this tensor's data
        kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_data, size / sizeof(float));
        cudaDeviceSynchronize();

        return shared_from_this();
    }
}
