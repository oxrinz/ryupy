#include "CudaTensor.h"
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace ryupy
{
    namespace cuda
    {
        CudaTensor::CudaTensor(const py::object &py_data) : Tensor(py_data)
        {
            std::vector<float> hostData = flattenData(py_data);

            size = hostData.size() * sizeof(float);

            cudaMalloc(&d_data, hostData.size() * sizeof(float));

            cudaMemcpy(d_data, hostData.data(), size, cudaMemcpyHostToDevice);
        }

        py::object CudaTensor::getFlattenedData() const
        {
            std::vector<float> hostData(size / sizeof(float));
            cudaMemcpy(hostData.data(), d_data, size, cudaMemcpyDeviceToHost);
            return py::cast(hostData);
        }

        py::object CudaTensor::getData() const
        {
            std::vector<float> hostData(size / sizeof(float));
            cudaMemcpy(hostData.data(), d_data, size, cudaMemcpyDeviceToHost);
            int index = 0;
            return reshapeData(hostData, shape, index);
        }

        CudaTensor::~CudaTensor()
        {
            cudaFree(d_data);
        }
    }
}
