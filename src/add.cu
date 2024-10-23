#include <cuda_runtime.h>

__global__ void addKernel(int* result, int a, int b) {
    *result = a + b;
}

extern "C" __declspec(dllexport) int add(int a, int b) {
    int result;  
    int* d_result; 

    cudaMalloc(&d_result, sizeof(int));

    addKernel<<<1, 1>>>(d_result, a, b);

    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_result);

    return result;
}
