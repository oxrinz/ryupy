#include "../Tensor.h"
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "kernels/Kernels.h"

namespace ryupy
{
    std::shared_ptr<Tensor> Tensor::negate()
    {
        std::shared_ptr<Tensor> tensor = handleInPlaceEmptyOperator(negateKernel);

        return tensor;
    }

    std::shared_ptr<Tensor> Tensor::log() const
    {
        std::shared_ptr<Tensor> tensor = handleEmptyOperator(logKernel);
        return tensor;
    }
}