#include <cuda_runtime.h>
#include <stdio.h>

namespace ryupy
{
        __global__ void equalityKernel(const float *a, const float *b, float *result, int size)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index < size)
            {
                result[index] = (a[index] == b[index]) ? 1.0f : 0.0f;
            }
        }

        __global__ void inequalityKernel(const float *a, const float *b, float *result, int size)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index < size)
            {
                result[index] = (a[index] != b[index]) ? 1.0f : 0.0f;
            }
        }

        __global__ void lessThanKernel(const float *a, const float *b, float *result, int size)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index < size)
            {
                result[index] = (a[index] < b[index]) ? 1.0f : 0.0f;
            }
        }

        __global__ void lessThanOrEqualKernel(const float *a, const float *b, float *result, int size)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index < size)
            {
                result[index] = (a[index] <= b[index]) ? 1.0f : 0.0f;
            }
        }

        __global__ void greaterThanKernel(const float *a, const float *b, float *result, int size)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index < size)
            {
                result[index] = (a[index] > b[index]) ? 1.0f : 0.0f;
            }
        }

        __global__ void greaterThanOrEqualKernel(const float *a, const float *b, float *result, int size)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index < size)
            {
                result[index] = (a[index] >= b[index]) ? 1.0f : 0.0f;
            }
        }
}
