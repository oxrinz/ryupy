#include "../Tensor.h"
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "kernels/Kernels.h"

namespace ryupy
{
    std::shared_ptr<Tensor> Tensor::operator&(Tensor &other)
    {
        return handleOperator(other, bitwiseAndKernel);
    }

    std::shared_ptr<Tensor> Tensor::operator|(Tensor &other)
    {
        return handleOperator(other, bitwiseOrKernel);
    }

    std::shared_ptr<Tensor> Tensor::operator^(Tensor &other)
    {
        return handleOperator(other, bitwiseXorKernel);
    }

    std::shared_ptr<Tensor> Tensor::operator~() const
    {
        return handleEmptyOperator(bitwiseNotKernel);
    }

    std::shared_ptr<Tensor> Tensor::operator<<(int shift) const
    {
        return handleShiftOperator(shift, leftShiftKernel);
    }

    std::shared_ptr<Tensor> Tensor::operator>>(int shift) const
    {
        return handleShiftOperator(shift, rightShiftKernel);
    }

    std::shared_ptr<Tensor> Tensor::operator&=(const Tensor &other)
    {
        return handleInPlaceOperator(other, bitwiseAndKernel);
    }

    std::shared_ptr<Tensor> Tensor::operator|=(const Tensor &other)
    {
        return handleInPlaceOperator(other, bitwiseOrKernel);
    }

    std::shared_ptr<Tensor> Tensor::operator^=(const Tensor &other)
    {
        return handleInPlaceOperator(other, bitwiseXorKernel);
    }

    std::shared_ptr<Tensor> Tensor::operator<<=(int shift)
    {
        return handleInPlaceShiftOperator(shift, leftShiftKernel);
    }

    std::shared_ptr<Tensor> Tensor::operator>>=(int shift)
    {
        return handleInPlaceShiftOperator(shift, rightShiftKernel);
    }
}
