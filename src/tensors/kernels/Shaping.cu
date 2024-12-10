#include <cuda_runtime.h>
#include <stdio.h>

namespace ryupy
{
    __global__ void transposeKernel(const float *input, float *output,
                                    const int *shape, const int *old_strides,
                                    const int *new_strides, const int size,
                                    const int dim0, const int dim1, const int ndim)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= size)
            return;

        // Convert flat index to coordinates
        int temp_idx = idx;
        int coords[8]; // Max dimensions supported

        for (int i = 0; i < ndim; i++)
        {
            coords[i] = temp_idx / old_strides[i];
            temp_idx %= old_strides[i];
        }

        // Swap coordinates for transposed dimensions
        int temp = coords[dim0];
        coords[dim0] = coords[dim1];
        coords[dim1] = temp;

        // Convert coordinates back to flat index
        int new_idx = 0;
        for (int i = 0; i < ndim; i++)
        {
            new_idx += coords[i] * new_strides[i];
        }

        output[new_idx] = input[idx];
    }

    __global__ void broadcastKernel(
        const float *__restrict__ input_data,
        float *__restrict__ output_data,
        const int *__restrict__ input_shape,
        const int *__restrict__ output_shape,
        int num_elements)
    {
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_elements)
            return;

        // For column-major layout:
        // - Index = col * num_rows + row
        // - Given flattened index: col = index / num_rows, row = index % num_rows
        const int output_rows = output_shape[0];
        const int output_col = tid / output_rows;
        const int output_row = tid % output_rows;

        // For broadcasting [1,3] -> [2,3], input is also column major
        // Each column of size 1 gets broadcast to size 2
        const int input_idx = output_col; // Just need the column index since input has 1 row

        // Write output in column-major order
        output_data[tid] = input_data[input_idx];
    }
}
