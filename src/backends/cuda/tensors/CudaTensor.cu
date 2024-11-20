#include "CudaTensor.h"
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cudnn.h>
#include <numeric>

namespace ryupy
{
    namespace cuda
    {
        CudaTensor::CudaTensor(const py::object &py_data) : Tensor(py_data)
        {
            std::vector<float> hostData = flattenPythonData(py_data);

            size = hostData.size() * sizeof(float);

            cudaMalloc(&d_data, size);

            cudaMemcpy(d_data, hostData.data(), size, cudaMemcpyHostToDevice);

            cudnnCreateTensorDescriptor(&tensor_desc);

            int nbDims = shape.size();
            std::vector<int> strideA(nbDims);

            strideA[nbDims - 1] = 1;
            for (int i = nbDims - 2; i >= 0; --i)
            {
                strideA[i] = strideA[i + 1] * shape[i + 1];
            }

            cudnnSetTensorNdDescriptor(tensor_desc,
                                       CUDNN_DATA_FLOAT,
                                       nbDims,
                                       shape.data(),
                                       strideA.data());
        }

        CudaTensor::CudaTensor(std::vector<int> shape) : Tensor()
        {
            size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) * sizeof(float);

            std::cout << "init sexing " << size << std::endl;

            this->size = size;
            this->shape = shape;

            cudaMalloc(&d_data, size);

            cudnnCreateTensorDescriptor(&tensor_desc);

            int nbDims = shape.size();
            std::vector<int> strideA(nbDims);

            strideA[nbDims - 1] = 1;
            for (int i = nbDims - 2; i >= 0; --i)
            {
                strideA[i] = strideA[i + 1] * shape[i + 1];
            }

            cudnnSetTensorNdDescriptor(tensor_desc,
                                       CUDNN_DATA_FLOAT,
                                       nbDims,
                                       shape.data(),
                                       strideA.data());
        }

        CudaTensor::CudaTensor(int size, std::vector<int> shape) : Tensor()
        {
            this->size = size;
            this->shape = shape;

            cudaMalloc(&d_data, size);

            cudnnCreateTensorDescriptor(&tensor_desc);

            int nbDims = shape.size();
            std::vector<int> strideA(nbDims);

            strideA[nbDims - 1] = 1;
            for (int i = nbDims - 2; i >= 0; --i)
            {
                strideA[i] = strideA[i + 1] * shape[i + 1];
            }

            cudnnSetTensorNdDescriptor(tensor_desc,
                                       CUDNN_DATA_FLOAT,
                                       nbDims,
                                       shape.data(),
                                       strideA.data());
        }
 
        py::object CudaTensor::getFlattenedData() const
        {
            std::vector<float> hostData(size / sizeof(float));
            cudaMemcpy(hostData.data(), d_data, size, cudaMemcpyDeviceToHost);
            return py::cast(hostData);
        }

        py::object CudaTensor::getData() const
        {
            std::cout << "sex " << size << std::endl;
            std::vector<float> hostData(size / sizeof(float));
            cudaMemcpy(hostData.data(), d_data, size, cudaMemcpyDeviceToHost);
            return reshapeData(hostData, shape);
        }

        CudaTensor::~CudaTensor()
        {
            cudnnDestroyTensorDescriptor(tensor_desc);
            cudaFree(d_data);
        }
    }
}
