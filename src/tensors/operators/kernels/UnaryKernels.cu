#include <cuda_runtime.h>
#include <stdio.h>

namespace ryupy
{
    __global__ void logKernel(const float *input, float *output, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            output[idx] = logf(input[idx]);
        }
    }

    __global__ void negateKernel(const float *input, float *output, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            output[idx] = -input[idx];
        }
    }
}