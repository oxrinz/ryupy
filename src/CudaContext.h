#pragma once

#include <cudnn.h>
#include <cublas_v2.h>
#include <memory>

namespace ryupy
{
    class CUDAContext
    {
    private:
        static CUDAContext *instance;
        cudnnHandle_t cudnn_handle;
        cublasHandle_t cublas_handle;

        // Private constructor prevents direct instantiation
        CUDAContext()
        {
            cudnnCreate(&cudnn_handle);
            cublasCreate(&cublas_handle);
        }

    public:
        // Delete copy constructor and assignment
        CUDAContext(const CUDAContext &) = delete;
        CUDAContext &operator=(const CUDAContext &) = delete;

        ~CUDAContext()
        {
            if (cudnn_handle)
            {
                cudnnDestroy(cudnn_handle);
            }
            if (cublas_handle)
            {
                cublasDestroy(cublas_handle);
            }
        }

        static CUDAContext &getInstance()
        {
            if (!instance)
            {
                instance = new CUDAContext();
            }
            return *instance;
        }

        cudnnHandle_t getCudnnHandle() { return cudnn_handle; }
        cublasHandle_t getCublasHandle() { return cublas_handle; }

        static void cleanup()
        {
            if (instance)
            {
                delete instance;
                instance = nullptr;
            }
        }
    };

    // In the corresponding cpp file (cuda_context.cpp)
    CUDAContext *CUDAContext::instance = nullptr;
}