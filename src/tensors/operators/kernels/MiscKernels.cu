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

    __global__ void transpose_kernel(const float *input, float *output,
                                     const int *input_shape, const int *input_strides,
                                     const int *output_strides, const int *perm,
                                     int ndim, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= size)
            return;

        int coords[32];
        int in_idx = idx;
        int out_idx = 0;

        for (int i = 0; i < ndim; i++)
        {
            coords[i] = in_idx / input_strides[i];
            in_idx = in_idx % input_strides[i];
        }

        for (int i = 0; i < ndim; i++)
        {
            out_idx += coords[perm[i]] * output_strides[i];
        }

        output[out_idx] = input[idx];
    }
}