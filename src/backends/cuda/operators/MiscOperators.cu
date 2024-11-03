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
            if (shape.size() < 2 || other.shape.size() < 2)
            {
                throw std::invalid_argument("Both tensors must have at least 2 dimensions for matrix multiplication.");
            }

            int batchSize = 1;
            int minBatchDim = std::min(shape.size(), other.shape.size()) - 2;

            for (int i = 0; i < minBatchDim; ++i)
            {
                if (shape[i] != other.shape[i])
                {
                    throw std::invalid_argument("Batch dimensions must match for matrix multiplication.");
                }
                batchSize *= shape[i];
            }

            int m = shape[shape.size() - 2];
            int n = shape[shape.size() - 1];
            int n_other = other.shape[other.shape.size() - 2];
            int p = other.shape[other.shape.size() - 1];

            if (n != n_other)
            {
                throw std::invalid_argument("Inner dimensions must match for matrix multiplication.");
            }

            auto result = std::make_shared<CudaTensor>(*this);

            cudaMalloc(&result->d_data, batchSize * m * p * sizeof(float));

            int lda = n;
            int ldb = p;
            int ldc = p;
            int strideA = m * n;
            int strideB = n * p;
            int strideC = m * p;

            cublasHandle_t handle;
            cublasCreate(&handle);

            const float alpha = 1.0f;
            const float beta = 0.0f;

            cublasSgemmStridedBatched(handle,
                                      CUBLAS_OP_N, CUBLAS_OP_N,
                                      m, p, n,
                                      &alpha,
                                      d_data, lda, strideA,
                                      other.d_data, ldb, strideB,
                                      &beta,
                                      result->d_data, ldc, strideC,
                                      batchSize);

            cublasDestroy(handle);

            return result;
        }
    }
}
