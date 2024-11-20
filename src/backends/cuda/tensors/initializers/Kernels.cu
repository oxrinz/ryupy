#include <cuda_runtime.h>
#include <stdio.h>

namespace ryupy
{
    namespace cuda
    {
        __global__ void zerosKernel(float *result, int size)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index < size)
            {
                result[index] = 0;
            }
        }

        __global__ void onesKernel(float *result, int size)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index < size)
            {
                result[index] = 1;
            }
        }

        __global__ void fillKernel(float *result, float val, int size)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index < size)
            {
                result[index] = val;
            }
        }

        __global__ void arangeKernel(float *data, float start, float step, int size)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size)
            {
                data[idx] = start + idx * step;
            }
        }

        __global__ void linspaceKernel(float *data, float start, float step, int size)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size)
            {
                data[idx] = start + idx * step;
            }
        }

        __global__ void eyeKernel(float *data, int n)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n * n)
            {
                int row = idx / n;
                int col = idx % n;
                data[idx] = (row == col) ? 1.0f : 0.0f;
            }
        }

        __global__ void scaleKernel(float *data, float offset, float scale, int size)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size)
            {
                // Transform from [0,1] to [low,high] using: value * (high-low) + low
                // where scale = (high-low) and offset = low
                data[idx] = data[idx] * scale + offset;
            }
        }
    }
}
