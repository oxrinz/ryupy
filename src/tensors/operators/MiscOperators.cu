#include "../Tensor.h"
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "kernels/Kernels.h"
#include <cublas_v2.h>
#include <memory>
#include <numeric>

namespace ryupy
{
    std::shared_ptr<Tensor> Tensor::pow(const Tensor &other) const
    {
        return handleOperator(other, powerKernel);
    }

    std::shared_ptr<Tensor> Tensor::ipow(const Tensor &other)
    {
        return handleInPlaceOperator(other, powerKernel);
    }

    std::shared_ptr<Tensor> Tensor::matmul(const Tensor &other) const
    {
        if (shape.size() < 2 || other.shape.size() < 2)
        {
            throw std::invalid_argument("Both tensors must have at least 2 dimensions for matrix multiplication.");
        }

        int batchSize = 1;
        int batchDim = shape.size() - 2;

        if (shape.size() != other.shape.size())
        {
            throw std::invalid_argument("Input tensors must have the same number of dimensions.");
        }

        for (int i = 0; i < batchDim; ++i)
        {
            if (shape[i] != other.shape[i])
            {
                throw std::invalid_argument("Batch dimensions must match for matrix multiplication.");
            }
            batchSize *= shape[i];
        }

        int m = shape[shape.size() - 2];
        int k = shape[shape.size() - 1];
        int k_other = other.shape[other.shape.size() - 2];
        int n = other.shape[other.shape.size() - 1];

        if (k != k_other)
        {
            throw std::invalid_argument("Inner dimensions must match for matrix multiplication.");
        }

        std::vector<int> result_shape(shape.begin(), shape.begin() + batchDim);
        result_shape.push_back(m);
        result_shape.push_back(n);

        size_t total_elements = batchSize * m * n;

        std::shared_ptr<Tensor> result = std::make_shared<Tensor>(total_elements * sizeof(float), result_shape);

        int lda = k;
        int ldb = k;
        int ldc = n;

        long long int strideA = static_cast<long long int>(m) * k;
        long long int strideB = static_cast<long long int>(k) * n;
        long long int strideC = static_cast<long long int>(m) * n;

        cublasHandle_t handle;
        cublasCreate(&handle);

        const float alpha = 1.0f;
        const float beta = 0.0f;

        cublasStatus_t status = cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, k,
            &alpha,
            d_data, lda, strideA,
            other.d_data, ldb, strideB,
            &beta,
            result->d_data, ldc, strideC,
            batchSize);

        if (status != CUBLAS_STATUS_SUCCESS)
        {
            cublasDestroy(handle);
            throw std::runtime_error("cublasSgemmStridedBatched failed");
        }

        cublasDestroy(handle);

        return result;
    }
}
