#include "Tensor.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>

namespace ryupy
{
    std::pair<int, int> Tensor::calculate_fans(const std::vector<int> &shape)
    {
        if (shape.size() < 2)
        {
            return {1, 1};
        }

        int fan_in = shape[shape.size() - 1];
        int fan_out = shape[shape.size() - 2];

        if (shape.size() > 2)
        {
            for (size_t i = 0; i < shape.size() - 2; i++)
            {
                fan_in *= shape[i];
                fan_out *= shape[i];
            }
        }

        return {fan_in, fan_out};
    }

    std::shared_ptr<Tensor> Tensor::copy() const
    {
        auto result = std::make_shared<Tensor>(shape);

        cudaMemcpy(result->d_data, d_data, size, cudaMemcpyDeviceToDevice);

        result->requires_grad = requires_grad;
        result->is_leaf = is_leaf;
        result->size = size;

        int nbDims = shape.size();
        std::vector<int> strideA(nbDims);
        strideA[nbDims - 1] = 1;
        for (int i = nbDims - 2; i >= 0; --i)
        {
            strideA[i] = strideA[i + 1] * shape[i + 1];
        }
        cudnnSetTensorNdDescriptor(result->tensor_desc,
                                   CUDNN_DATA_FLOAT,
                                   nbDims,
                                   shape.data(),
                                   strideA.data());

        return result;
    }

    std::vector<int> Tensor::calculate_strides(const std::vector<int> &shape) const
    {
        std::vector<int> strides(shape.size());
        int stride = 1;
        for (int i = shape.size() - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }
}
