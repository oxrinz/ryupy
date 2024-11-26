#include "../../Tensor.h"
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../../operators/kernels/Kernels.h"
#include <iostream>

namespace ryupy
{
    void Tensor::addBackward()
    {
        auto input1 = prev[0];
        auto input2 = prev[1];

        if (input1->requires_grad)
        {
            input1->grad = grad->copy();
        }

        if (input2->requires_grad)
        {
            input2->grad = grad->copy();
        }
    }

    void Tensor::subtractBackward()
    {
        auto input1 = prev[0];
        auto input2 = prev[1];

        if (input1->requires_grad)
        {
            input1->grad = grad->copy();
        }

        if (input2->requires_grad)
        {
            input2->grad = grad->copy()->negate();
        }
    }

    void Tensor::multiplyBackward()
    {
        auto input1 = prev[0];
        auto input2 = prev[1];

        if (input1->requires_grad)
        {
            input1->grad = grad->copy()->operator*(*input2);
        }
        if (input2->requires_grad)
        {
            input2->grad = grad->copy()->operator*(*input1);
        }
    }

    void Tensor::divideBackward()
    {
        auto input1 = prev[0];
        auto input2 = prev[1];

        if (input1->requires_grad)
        {
            input1->grad = grad->copy()->operator/(*input2);
        }
        if (input2->requires_grad)
        {
            auto temp = input1->operator/(*input2->operator*(*input2));
            input2->grad = grad->copy()->operator*(*temp)->negate();
        }
    }
}