#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdlib>
#include <vector>
#include <iostream>

__global__ void kernel_vec_add(float *A, float *B, float *C) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

__global__ void kernel_vec_mult(float *A, float *B, float *C) {
    int i = threadIdx.x;
    C[i] = A[i] * B[i];
}

std::vector<float> vec_operation(const std::vector<float> &A, const std::vector<float> &B, void(*kernel)(float*, float*, float*)) {
    int N = A.size();

    if (N != B.size()) {
        std::cerr << "Error: Vectors must have the same size!" << std::endl;
        return std::vector<float>();
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    cudaMemcpy(d_A, A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    kernel<<<1, N>>>(d_A, d_B, d_C); 

    std::vector<float> C(N);
    cudaMemcpy(C.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

std::vector<float> vec_add(const std::vector<float> &A, const std::vector<float> &B) {
    return vec_operation(A, B, kernel_vec_add);
}

std::vector<float> vec_mult(const std::vector<float> &A, const std::vector<float> &B) {
    return vec_operation(A, B, kernel_vec_mult);
}

void register_math_submodule(pybind11::module& m) {
    pybind11::module math_submodule = m.def_submodule("math");
    math_submodule.def("vec_add", &vec_add, "Function for vector addition");
    math_submodule.def("vec_mult", &vec_mult, "Function for vector addition");
}