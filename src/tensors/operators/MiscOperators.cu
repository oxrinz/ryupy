#include "../Tensor.h"
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "kernels/Kernels.h"
#include <cublas_v2.h>
#include <memory>
#include <numeric>
#include <iostream>

namespace ryupy
{

    std::shared_ptr<Tensor> Tensor::pow(Tensor &other)
    {
        return handleOperator(other, powerKernel, &Tensor::powerBackward);
    }

    std::shared_ptr<Tensor> Tensor::ipow(Tensor &other)
    {
        return handleInPlaceOperator(other, powerKernel);
    }

    std::shared_ptr<Tensor> Tensor::matmul(Tensor &other)
    {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasHandle_t handle;

        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error("Failed to create cuBLAS handle");
        }

        std::unique_ptr<std::remove_pointer<cublasHandle_t>::type, decltype(&cublasDestroy)>
            handle_guard(handle, cublasDestroy);

        if (shape.size() == 1 && other.shape.size() == 2)
        {
            if (shape[0] != other.shape[0])
            {
                throw std::invalid_argument("Vector dimension must match matrix's first dimension.");
            }

            std::vector<int> result_shape = {other.shape[1]};
            auto result = std::make_shared<Tensor>(other.shape[1] * sizeof(float), result_shape);

            if (requires_grad || other.requires_grad)
            {
                result->requires_grad = true;
                result->is_leaf = false;
                result->prev.push_back(shared_from_this());
                result->prev.push_back(other.shared_from_this());
                result->backward_fn = [result]()
                { result.get()->matmulBackward(); };
            }

            if (cublasSgemv(handle, CUBLAS_OP_T,
                            other.shape[0], other.shape[1],
                            &alpha,
                            other.d_data, other.shape[0],
                            d_data, 1,
                            &beta,
                            result->d_data, 1) != CUBLAS_STATUS_SUCCESS)
            {
                throw std::runtime_error("cublasSgemv failed in vector-matrix multiplication");
            }
            return result;
        }

        if (shape.size() == 2 && other.shape.size() == 1)
        {
            if (shape[1] != other.shape[0])
            {
                throw std::invalid_argument("Matrix second dimension must match vector dimension.");
            }

            std::vector<int> result_shape = {shape[0]};
            auto result = std::make_shared<Tensor>(shape[0] * sizeof(float), result_shape);

            if (requires_grad || other.requires_grad)
            {
                result->requires_grad = true;
                result->is_leaf = false;
                result->prev.push_back(shared_from_this());
                result->prev.push_back(other.shared_from_this());
                result->backward_fn = [result]()
                { result.get()->matmulBackward(); };
            }

            if (cublasSgemv(handle, CUBLAS_OP_N,
                            shape[0], shape[1],
                            &alpha,
                            d_data, shape[1],
                            other.d_data, 1,
                            &beta,
                            result->d_data, 1) != CUBLAS_STATUS_SUCCESS)
            {
                throw std::runtime_error("cublasSgemv failed in matrix-vector multiplication");
            }
            return result;
        }

        if (shape.size() >= 2 && other.shape.size() >= 2)
        {
            if (shape.size() != other.shape.size())
            {
                throw std::invalid_argument("Input tensors must have the same number of dimensions.");
            }

            int batchSize = 1;
            int batchDim = shape.size() - 2;

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

            auto result = std::make_shared<Tensor>(batchSize * m * n * sizeof(float), result_shape);

            if (requires_grad || other.requires_grad)
            {
                result->requires_grad = true;
                result->is_leaf = false;
                result->prev.push_back(shared_from_this());
                result->prev.push_back(other.shared_from_this());
                result->backward_fn = [result]()
                { result.get()->matmulBackward(); };
            }

            std::cout << "m: " << m << " n: " << n << " k: " << k << " batchSize: " << batchSize << std::endl;

            if (cublasSgemmStridedBatched(handle,
                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                          n, m, k,
                                          &alpha,
                                          other.d_data, n, k * n, // Matrix B
                                          d_data, k, m * k,       // Matrix A
                                          &beta,
                                          result->d_data, n, m * n,
                                          batchSize) != CUBLAS_STATUS_SUCCESS)
            {
                throw std::runtime_error("cublasSgemmStridedBatched failed");
            }
            return result;
        }

        throw std::invalid_argument("Invalid tensor dimensions for matrix multiplication");
    }
}