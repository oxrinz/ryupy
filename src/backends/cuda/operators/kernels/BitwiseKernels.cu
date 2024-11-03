#include <cuda_runtime.h>
#include <stdio.h>

namespace ryupy
{
    namespace cuda
    {
        __global__ void bitwiseAndKernel(const float *a, const float *b, float *result, int size)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index < size)
            {
                result[index] = static_cast<int>(a[index]) & static_cast<int>(b[index]);
            }
        }

        __global__ void bitwiseOrKernel(const float *a, const float *b, float *result, int size)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index < size)
            {
                result[index] = static_cast<int>(a[index]) | static_cast<int>(b[index]);
            }
        }

        __global__ void bitwiseXorKernel(const float *a, const float *b, float *result, int size)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index < size)
            {
                result[index] = static_cast<int>(a[index]) ^ static_cast<int>(b[index]);
            }
        }

        __global__ void bitwiseNotKernel(const float *a, float *result, int size)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index < size)
            {
                result[index] = ~static_cast<int>(a[index]);
            }
        }

        __global__ void leftShiftKernel(const float* a, float* result, int size, int shift)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index < size)
            {
                result[index] = static_cast<int>(a[index]) << shift;
            }
        }

        __global__ void rightShiftKernel(const float* a, float* result, int size, int shift)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index < size)
            {
                result[index] = static_cast<int>(a[index]) >> shift;
            }
        }
    }
}
