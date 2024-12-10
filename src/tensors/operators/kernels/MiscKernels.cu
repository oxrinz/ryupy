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


    __global__ void permuteDimsKernel(const float *in, float *out, int *in_shape, int *out_shape, int *perm, int ndim, int total)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < total)
        {
            int coords_in[32];
            int tmp = idx;
            for (int i = ndim - 1; i >= 0; i--)
            {
                int s = out_shape[i];
                coords_in[i] = tmp % s;
                tmp /= s;
            }
            int coords_orig[32];
            for (int i = 0; i < ndim; i++)
            {
                coords_orig[perm[i]] = coords_in[i];
            }
            int stride = 1;
            int in_idx = 0;
            for (int i = ndim - 1; i >= 0; i--)
            {
                in_idx += coords_orig[i] * stride;
                stride *= in_shape[i];
            }
            out[idx] = in[in_idx];
        }
    }
}