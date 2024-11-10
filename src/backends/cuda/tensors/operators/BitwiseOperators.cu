#include "../CudaTensor.h"
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "kernels/Kernels.h"

namespace ryupy
{
    namespace cuda
    {
        std::shared_ptr<CudaTensor> CudaTensor::operator&(const CudaTensor &other) const
        {
            return handleOperator(other, bitwiseAndKernel);
        }

        std::shared_ptr<CudaTensor> CudaTensor::operator|(const CudaTensor &other) const
        {
            return handleOperator(other, bitwiseOrKernel);
        }

        std::shared_ptr<CudaTensor> CudaTensor::operator^(const CudaTensor &other) const
        {
            return handleOperator(other, bitwiseXorKernel);
        }

        std::shared_ptr<CudaTensor> CudaTensor::operator~() const
        {
            return handleEmptyOperator(bitwiseNotKernel);
        }

        std::shared_ptr<CudaTensor> CudaTensor::operator<<(int shift) const
        {
            return handleShiftOperator(shift, leftShiftKernel);
        }

        std::shared_ptr<CudaTensor> CudaTensor::operator>>(int shift) const
        {
            return handleShiftOperator(shift, rightShiftKernel);
        }



        std::shared_ptr<CudaTensor> CudaTensor::operator&=(const CudaTensor &other)
        {
            return handleInPlaceOperator(other, bitwiseAndKernel);
        }

        std::shared_ptr<CudaTensor> CudaTensor::operator|=(const CudaTensor &other)
        {
            return handleInPlaceOperator(other, bitwiseOrKernel);
        }

        std::shared_ptr<CudaTensor> CudaTensor::operator^=(const CudaTensor &other)
        {
            return handleInPlaceOperator(other, bitwiseXorKernel);
        }

        std::shared_ptr<CudaTensor> CudaTensor::operator<<=(int shift)
        {
            return handleInPlaceShiftOperator(shift, leftShiftKernel);
        }

        std::shared_ptr<CudaTensor> CudaTensor::operator>>=(int shift)
        {
            return handleInPlaceShiftOperator(shift, rightShiftKernel);
        }
    }
}
