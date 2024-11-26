#include "../Tensor.h"
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "kernels/Kernels.h"

namespace ryupy
{
    std::shared_ptr<Tensor> Tensor::operator+(Tensor &other)
    {
        return handleOperator(other, addKernel, &Tensor::addBackward);
    }

    std::shared_ptr<Tensor> Tensor::operator-(Tensor &other)
    {
        return handleOperator(other, subtractKernel, &Tensor::subtractBackward);
    }

    std::shared_ptr<Tensor> Tensor::operator*(Tensor &other)
    {
        return handleOperator(other, multiplyKernel, &Tensor::multiplyBackward);
    }

    std::shared_ptr<Tensor> Tensor::operator/(Tensor &other)
    {
        return handleOperator(other, divideKernel, &Tensor::divideBackward);
    }

    std::shared_ptr<Tensor> Tensor::operator%(Tensor &other)
    {
        return handleOperator(other, moduloKernel);
    }

    std::shared_ptr<Tensor> Tensor::operator+=(Tensor &other)
    {
        return handleInPlaceOperator(other, addKernel);
    }

    std::shared_ptr<Tensor> Tensor::operator-=(Tensor &other)
    {
        return handleInPlaceOperator(other, subtractKernel);
    }

    std::shared_ptr<Tensor> Tensor::operator*=(Tensor &other)
    {
        return handleInPlaceOperator(other, multiplyKernel);
    }

    std::shared_ptr<Tensor> Tensor::operator/=(Tensor &other)
    {
        return handleInPlaceOperator(other, divideKernel);
    }

    std::shared_ptr<Tensor> Tensor::operator%=(Tensor &other)
    {
        return handleInPlaceOperator(other, moduloKernel);
    }
}
