#include "../../Tensor.h"
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../../operators/kernels/Kernels.h"

namespace ryupy
{
    void Tensor::addBackward()
    {
        auto input1 = prev[0];
        auto input2 = prev[1];

        if (input1->requires_grad)
        {
            input1->grad = std::make_shared<Tensor>(input1->shape);

            input1->grad->handleInPlaceOperator(*grad, addKernel);
        }

        if (input2->requires_grad)
        {
            input2->grad = std::make_shared<Tensor>(input2->shape);

            input2->grad->handleInPlaceOperator(*grad, addKernel);
        }
    }
}