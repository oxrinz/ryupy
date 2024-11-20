#include "../Tensor.h"
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "kernels/Kernels.h"

namespace ryupy
{
    std::shared_ptr<Tensor> Tensor::operator+(const Tensor &other) const
    {
        return handleOperator(other, addKernel);
    }

    std::shared_ptr<Tensor> Tensor::operator-(const Tensor &other) const
    {
        return handleOperator(other, subtractKernel);
    }

    std::shared_ptr<Tensor> Tensor::operator*(const Tensor &other) const
    {
        return handleOperator(other, multiplyKernel);
    }

    std::shared_ptr<Tensor> Tensor::operator/(const Tensor &other) const
    {
        return handleOperator(other, divideKernel);
    }

    std::shared_ptr<Tensor> Tensor::operator%(const Tensor &other) const
    {
        return handleOperator(other, moduloKernel);
    }

    std::shared_ptr<Tensor> Tensor::operator+=(const Tensor &other)
    {
        return handleInPlaceOperator(other, addKernel);
    }

    std::shared_ptr<Tensor> Tensor::operator-=(const Tensor &other)
    {
        return handleInPlaceOperator(other, subtractKernel);
    }

    std::shared_ptr<Tensor> Tensor::operator*=(const Tensor &other)
    {
        return handleInPlaceOperator(other, multiplyKernel);
    }

    std::shared_ptr<Tensor> Tensor::operator/=(const Tensor &other)
    {
        return handleInPlaceOperator(other, divideKernel);
    }

    std::shared_ptr<Tensor> Tensor::operator%=(const Tensor &other)
    {
        return handleInPlaceOperator(other, moduloKernel);
    }
}
