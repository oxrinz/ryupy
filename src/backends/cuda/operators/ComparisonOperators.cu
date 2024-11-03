#include "../CudaTensor.h"
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "kernels/Kernels.h"

namespace ryupy
{
    namespace cuda
    {
        std::shared_ptr<CudaTensor> CudaTensor::operator==(const CudaTensor &other) const
        {
            return handleOperator(other, equalityKernel);
        }

        std::shared_ptr<CudaTensor> CudaTensor::operator!=(const CudaTensor &other) const
        {
            return handleOperator(other, inequalityKernel);
        }

        std::shared_ptr<CudaTensor> CudaTensor::operator<(const CudaTensor &other) const
        {
            return handleOperator(other, lessThanKernel);
        }

        std::shared_ptr<CudaTensor> CudaTensor::operator<=(const CudaTensor &other) const
        {
            return handleOperator(other, lessThanOrEqualKernel);
        }

        std::shared_ptr<CudaTensor> CudaTensor::operator>(const CudaTensor &other) const
        {
            return handleOperator(other, greaterThanKernel);
        }

        std::shared_ptr<CudaTensor> CudaTensor::operator>=(const CudaTensor &other) const
        {
            return handleOperator(other, greaterThanOrEqualKernel);
        }
    }
}
