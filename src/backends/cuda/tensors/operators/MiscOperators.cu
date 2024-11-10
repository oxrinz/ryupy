#include "../CudaTensor.h"
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "kernels/Kernels.h"
#include <cublas_v2.h>
#include <memory>
#include <numeric>

namespace ryupy
{
    namespace cuda
    {
        std::shared_ptr<CudaTensor> CudaTensor::pow(const CudaTensor &other) const
        {
            return handleOperator(other, powerKernel);
        }

        std::shared_ptr<CudaTensor> CudaTensor::ipow(const CudaTensor &other)
        {
            return handleInPlaceOperator(other, powerKernel);
        }

        std::shared_ptr<CudaTensor> CudaTensor::matmul(const CudaTensor &other) const
        {
            // Ensure both tensors have at least 2 dimensions
            if (shape.size() < 2 || other.shape.size() < 2)
            {
                throw std::invalid_argument("Both tensors must have at least 2 dimensions for matrix multiplication.");
            }

            // Calculate batch size and verify batch dimensions
            int batchSize = 1;
            int batchDim = shape.size() - 2; // Number of batch dimensions

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

            // Dimensions for matrix multiplication
            int m = shape[shape.size() - 2];                   // Rows in A
            int k = shape[shape.size() - 1];                   // Columns in A
            int k_other = other.shape[other.shape.size() - 2]; // Rows in B
            int n = other.shape[other.shape.size() - 1];       // Columns in B

            if (k != k_other)
            {
                throw std::invalid_argument("Inner dimensions must match for matrix multiplication.");
            }

            // Build result shape
            std::vector<int> result_shape(shape.begin(), shape.begin() + batchDim);
            result_shape.push_back(m);
            result_shape.push_back(n);

            // Allocate memory for result
            size_t total_elements = batchSize * m * n;

            // Prepare the result tensor
            std::shared_ptr<CudaTensor>  result = std::make_shared<CudaTensor>(total_elements * sizeof(float), result_shape);

            // cuBLAS parameters
            int lda = k; // Leading dimension of A
            int ldb = k; // Leading dimension of B
            int ldc = n; // Leading dimension of C

            long long int strideA = static_cast<long long int>(m) * k;
            long long int strideB = static_cast<long long int>(k) * n;
            long long int strideC = static_cast<long long int>(m) * n;

            cublasHandle_t handle;
            cublasCreate(&handle);

            const float alpha = 1.0f;
            const float beta = 0.0f;

            // Perform the matrix multiplication
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

            // Clean up
            cublasDestroy(handle);

            return result;
        }
    }
}
