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

    __global__ void sumKernel(const float *input, float *output, int size)
    {
        extern __shared__ float sdata[];

        unsigned int tid = threadIdx.x;
        unsigned int globalId = blockIdx.x * blockDim.x + threadIdx.x;

        // Initialize shared memory
        sdata[tid] = 0.0f;

        // Load and accumulate input into shared memory
        if (globalId < size)
        {
            sdata[tid] = input[globalId];
        }
        __syncthreads();

        // Do reduction in shared memory
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (tid < s && (tid + s) < blockDim.x)
            {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        // Write result for this block to global memory
        if (tid == 0)
        {
            atomicAdd(output, sdata[0]);
        }
    }

    __global__ void sumDimKernel(const float *input, float *output,
                                 const int *input_strides, const int *output_shape,
                                 const int *input_shape,
                                 int reduce_dim, int reduce_size, int output_size,
                                 int num_dims)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= output_size)
            return;

        // Calculate indices for output element
        int remaining_idx = idx;
        int indices[32]; // For each dimension
        int out_pos = 0;

        // Map output index to input coordinates
        for (int d = num_dims - 1; d >= 0; d--)
        {
            if (d == reduce_dim)
            {
                indices[d] = 0; // Will be used in reduction loop
                continue;
            }

            indices[d] = remaining_idx % input_shape[d];
            remaining_idx /= input_shape[d];
            out_pos++;
        }

        // Calculate base input index
        int input_idx = 0;
        for (int d = 0; d < num_dims; d++)
        {
            if (d != reduce_dim)
            {
                input_idx += indices[d] * input_strides[d];
            }
        }

        // Sum along reduction dimension
        float sum = 0.0f;
        for (int i = 0; i < reduce_size; i++)
        {
            sum += input[input_idx + i * input_strides[reduce_dim]];
        }
        output[idx] = sum;
    }
}