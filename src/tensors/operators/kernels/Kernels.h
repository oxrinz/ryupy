#pragma once

namespace ryupy
{
    __global__ void addKernel(const float *a, const float *b, float *result, int size);
    __global__ void subtractKernel(const float *a, const float *b, float *result, int size);
    __global__ void multiplyKernel(const float *a, const float *b, float *result, int size);
    __global__ void divideKernel(const float *a, const float *b, float *result, int size);
    __global__ void moduloKernel(const float *a, const float *b, float *result, int size);

    __global__ void powerKernel(const float *a, const float *b, float *result, int size);

    __global__ void equalityKernel(const float *a, const float *b, float *result, int size);
    __global__ void inequalityKernel(const float *a, const float *b, float *result, int size);
    __global__ void lessThanKernel(const float *a, const float *b, float *result, int size);
    __global__ void lessThanOrEqualKernel(const float *a, const float *b, float *result, int size);
    __global__ void greaterThanKernel(const float *a, const float *b, float *result, int size);
    __global__ void greaterThanOrEqualKernel(const float *a, const float *b, float *result, int size);

    __global__ void bitwiseAndKernel(const float *a, const float *b, float *result, int size);
    __global__ void bitwiseOrKernel(const float *a, const float *b, float *result, int size);
    __global__ void bitwiseXorKernel(const float *a, const float *b, float *result, int size);
    __global__ void bitwiseNotKernel(const float *a, float *result, int size);
    __global__ void leftShiftKernel(const float *a, float *result, int size, int shift);
    __global__ void rightShiftKernel(const float *a, float *result, int size, int shift);

    __global__ void logKernel(const float *input, float *output, int size);
    __global__ void negateKernel(const float *input, float *output, int size);
    __global__ void sumKernel(const float *input, float *output, int size);
    __global__ void sumDimKernel(const float *input, float *output,
                                 const int *input_strides, const int *output_shape,
                                 const int *input_shape,
                                 int reduce_dim, int reduce_size, int output_size,
                                 int num_dims);

    __global__ void permuteDimsKernel(const float *in, float *out, int *in_shape, int *out_shape, int *perm, int ndim, int total);

}
