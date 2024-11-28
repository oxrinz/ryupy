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

    __global__ void sumReduceKernel(const float *input, float *output, int size)
    {
        extern __shared__ float sdata[];
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

        sdata[tid] = 0;
        if (i < size)
        {
            sdata[tid] = input[i];
        }
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0)
        {
            atomicAdd(output, sdata[0]);
        }
    }
}