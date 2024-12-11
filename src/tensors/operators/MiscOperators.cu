#include "../Tensor.h"
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "kernels/Kernels.h"
#include "../../CudaContext.h"
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

    // In matmul function
    std::shared_ptr<Tensor> Tensor::matmul(Tensor &other)
    {
        // Handle vector-matrix case
        if (shape.size() == 1)
        {
            auto reshaped = reshape({1, shape[0]});
            reshaped->requires_grad = requires_grad;
            auto intermediate = reshaped->matmul(other);
            auto result = intermediate->reshape({intermediate->shape.back()});

            if (requires_grad || other.requires_grad)
            {
                result->requires_grad = true;
                result->is_leaf = false;
                result->prev = {shared_from_this(), other.shared_from_this()};
                result->backward_fn = [result]()
                { result->matmulBackward(); };
            }
            return result;
        }

        // Handle matrix-vector case
        if (other.shape.size() == 1)
        {
            auto reshaped = other.reshape({other.shape[0], 1});
            reshaped->requires_grad = other.requires_grad;
            auto intermediate = matmul(*reshaped);
            auto result = intermediate->reshape({intermediate->shape[0]});

            if (requires_grad || other.requires_grad)
            {
                result->requires_grad = true;
                result->is_leaf = false;
                result->prev = {shared_from_this(), other.shared_from_this()};
                result->backward_fn = [result]()
                { result->matmulBackward(); };
            }
            return result;
        }

        // Get matrix dimensions
        int m = shape[shape.size() - 2];
        int k = shape[shape.size() - 1];
        int n = other.shape[other.shape.size() - 1];

        if (k != other.shape[other.shape.size() - 2])
            throw std::runtime_error("Incompatible matrix dimensions for multiplication");

        // Get batch dimensions
        std::vector<int> batch_dims1(shape.begin(), shape.end() - 2);
        std::vector<int> batch_dims2(other.shape.begin(), other.shape.end() - 2);

        // Compute broadcast batch shape
        std::vector<int> broadcast_shape;
        int max_batch_dims = std::max(batch_dims1.size(), batch_dims2.size());

        while (batch_dims1.size() < max_batch_dims)
            batch_dims1.insert(batch_dims1.begin(), 1);
        while (batch_dims2.size() < max_batch_dims)
            batch_dims2.insert(batch_dims2.begin(), 1);

        for (size_t i = 0; i < max_batch_dims; i++)
        {
            if (batch_dims1[i] == batch_dims2[i])
                broadcast_shape.push_back(batch_dims1[i]);
            else if (batch_dims1[i] == 1)
                broadcast_shape.push_back(batch_dims2[i]);
            else if (batch_dims2[i] == 1)
                broadcast_shape.push_back(batch_dims1[i]);
            else
                throw std::runtime_error("Incompatible broadcast batch dimensions");
        }

        // Calculate result shape and create output tensor
        std::vector<int> result_shape = broadcast_shape;
        result_shape.push_back(m);
        result_shape.push_back(n);

        auto result = std::make_shared<Tensor>(result_shape);
        result->requires_grad = requires_grad || other.requires_grad;

        // Calculate total batch size
        int total_batch_size = 1;
        for (int dim : broadcast_shape)
            total_batch_size *= dim;

        float alpha = 1.0f;
        float beta = 0.0f;

        // Process each batch with broadcasting
        for (int batch = 0; batch < total_batch_size; batch++)
        {
            int offset1 = calculateBroadcastOffset(batch, batch_dims1, broadcast_shape) * m * k;
            int offset2 = calculateBroadcastOffset(batch, batch_dims2, broadcast_shape) * k * n;
            int result_offset = batch * m * n;

            cublasGemmEx(CUDAContext::getInstance().getCublasHandle(),
                         CUBLAS_OP_N, CUBLAS_OP_N,
                         n, m, k,
                         &alpha,
                         other.d_data + offset2, CUDA_R_32F, n,
                         d_data + offset1, CUDA_R_32F, k,
                         &beta,
                         result->d_data + result_offset, CUDA_R_32F, n,
                         CUDA_R_32F,
                         CUBLAS_GEMM_DEFAULT);
        }

        if (result->requires_grad)
        {
            result->is_leaf = false;
            result->prev = {shared_from_this(), other.shared_from_this()};
            result->backward_fn = [result]()
            { result->matmulBackward(); };
        }

        return result;
    }

}