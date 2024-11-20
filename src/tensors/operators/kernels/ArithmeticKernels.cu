#include <cuda_runtime.h>
#include <stdio.h>

namespace ryupy
{
    __global__ void addKernel(const float *a, const float *b, float *result, int size)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < size)
        {
            result[index] = a[index] + b[index];
        }
    }

    __global__ void subtractKernel(const float *a, const float *b, float *result, int size)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < size)
        {
            result[index] = a[index] - b[index];
        }
    }

    __global__ void multiplyKernel(const float *a, const float *b, float *result, int size)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < size)
        {
            result[index] = a[index] * b[index];
        }
    }

    __global__ void divideKernel(const float *a, const float *b, float *result, int size)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < size)
        {
            result[index] = a[index] / b[index];
        }
    }

    __global__ void moduloKernel(const float *a, const float *b, float *result, int size)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < size)
        {
            result[index] = fmod(a[index], b[index]);
        }
    }
}