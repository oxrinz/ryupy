#include "Tensor.h"
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cudnn.h>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <iostream>

#define RESET "\033[0m"
#define WHITE "\033[37m"
#define BLUE "\033[34m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define MAGENTA "\033[35m"
#define CYAN "\033[36m"
#define BRIGHT_WHITE "\033[97m"
#define BRIGHT_BLUE "\033[94m"
#define BRIGHT_RED "\033[91m"
#define BRIGHT_GREEN "\033[92m"
#define BRIGHT_YELLOW "\033[93m"
#define BRIGHT_MAGENTA "\033[95m"
#define BRIGHT_CYAN "\033[96m"

namespace ryupy
{
    Tensor::Tensor(std::vector<int> shape) : Tensor()
    {
        if (std::any_of(shape.begin(), shape.end(), [](int d)
                        { return d <= 0; }))
        {
            throw std::invalid_argument("All dimensions must be positive");
        }

        size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) * sizeof(float);
        this->size = size;
        this->shape = shape;

        cudaMalloc(&d_data, size);
        cudnnCreateTensorDescriptor(&tensor_desc);

        // For cuDNN, convert dimensions to NCHW format if needed
        int nbDims = shape.size();
        while (nbDims < 4)
        {
            shape.insert(shape.begin(), 1);
            nbDims++;
        }

        std::vector<int> stride(nbDims);
        stride[0] = 1;
        for (int i = 1; i < nbDims; i++)
        {
            stride[i] = stride[i - 1] * shape[i - 1];
        }

        cudnnStatus_t status = cudnnSetTensorNdDescriptor(
            tensor_desc,
            CUDNN_DATA_FLOAT,
            nbDims,
            shape.data(),
            stride.data());

        if (status != CUDNN_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::string("CUDNN error: ") + cudnnGetErrorString(status));
        }
    }

    Tensor::~Tensor()
    {
        if (d_data != nullptr)
        {
            cudaFree(d_data);
            d_data = nullptr;
        }
        if (tensor_desc != nullptr)
        {
            cudnnDestroyTensorDescriptor(tensor_desc);
            tensor_desc = nullptr;
        }
    }
}
