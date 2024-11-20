#pragma once

namespace ryupy
{
        __global__ void zerosKernel(float *result, int size);
        __global__ void onesKernel(float *result, int size);
        __global__ void fillKernel(float *result, float val, int size);
        __global__ void arangeKernel(float *result, float start, float step, int size);
        __global__ void linspaceKernel(float* data, float start, float step, int size);
        __global__ void eyeKernel(float* data, int n);
        __global__ void scaleKernel(float *data, float offset, float scale, int size);
}
