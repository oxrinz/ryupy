#pragma once

namespace ryupy
{
    __global__ void transposeKernel(const float *input, float *output,
                                    const int *shape, const int *old_strides,
                                    const int *new_strides, const int size,
                                    const int dim0, const int dim1, const int ndim);

    __global__ void broadcastKernel(
        const float *__restrict__ input_data,
        float *__restrict__ output_data,
        const int *__restrict__ input_shape,
        const int *__restrict__ output_shape,
        int num_elements);
}
