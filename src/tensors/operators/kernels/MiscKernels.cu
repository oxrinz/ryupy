#include <cuda_runtime.h>
#include <stdio.h>

namespace ryupy
{
    __global__ void powerKernel(const float *a, const float *b, float *result, int size)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < size)
        {
            result[index] = powf(a[index], b[index]);
        }
    }
}