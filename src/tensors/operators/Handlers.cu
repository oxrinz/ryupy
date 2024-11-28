#include "../Tensor.h"
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <optional>
#include <numeric>

namespace ryupy
{
    std::shared_ptr<Tensor> Tensor::handleOperator(Tensor &other, KernelFunc kernel, void (Tensor::*backward_function)())
    {
        if (shape != other.shape)
        {
            throw std::invalid_argument("Tensors must be same shape");
        }

        auto result = std::make_shared<Tensor>(shape);

        if (requires_grad || other.requires_grad)
        {
            if (backward_function != nullptr)
            {
                result->requires_grad = true;
                result->is_leaf = false;
                result->prev.push_back(shared_from_this());
                result->prev.push_back(other.shared_from_this());
                result->backward_fn = [result, backward_function]()
                {
                    (result.get()->*backward_function)();
                };
            }
        }

        int threadsPerBlock = 256;
        int blocksPerGrid = (size / sizeof(float) + threadsPerBlock - 1) / threadsPerBlock;

        kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, other.d_data, result->d_data, size / sizeof(float));

        cudaDeviceSynchronize();

        return result;
    }

    std::shared_ptr<Tensor> Tensor::handleInPlaceOperator(Tensor &other, KernelFunc kernel)
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
        auto result = std::make_shared<Tensor>(shape);

        int threadsPerBlock = 256;
        int blocksPerGrid = (size / sizeof(float) + threadsPerBlock - 1) / threadsPerBlock;

        kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, result->d_data, size / sizeof(float), shift);
        cudaDeviceSynchronize();

        return result;
    }

    std::shared_ptr<Tensor> Tensor::handleEmptyOperator(KernelEmptyFunc kernel) const
    {
        auto result = std::make_shared<Tensor>(shape);

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

        kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_data, size / sizeof(float), shift);
        cudaDeviceSynchronize();

        return shared_from_this();
    }

    std::shared_ptr<Tensor> Tensor::handleInPlaceEmptyOperator(KernelEmptyFunc kernel)
    {
        int threadsPerBlock = 256;
        int blocksPerGrid = (size / sizeof(float) + threadsPerBlock - 1) / threadsPerBlock;

        kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_data, size / sizeof(float));
        cudaDeviceSynchronize();

        return shared_from_this();
    }

    float Tensor::handleReduceOperator(KernelEmptyFunc kernel) const
    {
        float *d_result;
        cudaMalloc(&d_result, sizeof(float));
        cudaMemset(d_result, 0, sizeof(float));

        int threadsPerBlock = 256;
        int blocksPerGrid = (size / sizeof(float) + threadsPerBlock - 1) / threadsPerBlock;

        kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
            d_data,
            d_result,
            size / sizeof(float));

        float result;
        cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_result);
        return result;
    }
}
