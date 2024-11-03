#include "../CudaTensor.h"
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "kernels/Kernels.h"

namespace ryupy
{
    namespace cuda
    {
        std::shared_ptr<CudaTensor> CudaTensor::operator+(const CudaTensor &other) const
        {
            return handleOperator(other, addKernel);
        }

        std::shared_ptr<CudaTensor> CudaTensor::operator-(const CudaTensor &other) const
        {
            return handleOperator(other, subtractKernel);
        }

        std::shared_ptr<CudaTensor> CudaTensor::operator*(const CudaTensor &other) const
        {
            return handleOperator(other, multiplyKernel);
        }

        std::shared_ptr<CudaTensor> CudaTensor::operator/(const CudaTensor &other) const
        {
            return handleOperator(other, divideKernel);
        }

        std::shared_ptr<CudaTensor> CudaTensor::operator%(const CudaTensor &other) const
        {
            return handleOperator(other, moduloKernel);
        }

        std::shared_ptr<CudaTensor> CudaTensor::operator+=(const CudaTensor &other)
        {
            return handleInPlaceOperator(other, addKernel);
        }

        std::shared_ptr<CudaTensor> CudaTensor::operator-=(const CudaTensor &other)
        {
            return handleInPlaceOperator(other, subtractKernel);
        }

        std::shared_ptr<CudaTensor> CudaTensor::operator*=(const CudaTensor &other)
        {
            return handleInPlaceOperator(other, multiplyKernel);
        }

        std::shared_ptr<CudaTensor> CudaTensor::operator/=(const CudaTensor &other)
        {
            return handleInPlaceOperator(other, divideKernel);
        }

        std::shared_ptr<CudaTensor> CudaTensor::operator%=(const CudaTensor &other)
        {
            return handleInPlaceOperator(other, moduloKernel);
        }
    }
}
