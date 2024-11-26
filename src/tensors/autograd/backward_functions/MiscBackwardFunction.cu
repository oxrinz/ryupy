#include "../../Tensor.h"
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../../operators/kernels/Kernels.h"
#include <iostream>

namespace ryupy
{
    void Tensor::powerBackward()
    {
        auto input1 = prev[0];
        auto input2 = prev[1];

        if (input1->requires_grad)
        {
            auto one = Tensor::fill(input2->shape, 1.0f);
            auto y_minus_1 = input2->operator-(*one);
            auto base_pow = input1->pow(*y_minus_1);
            auto temp = base_pow->operator*(*input2);
            input1->grad = grad->copy()->operator*(*temp);
        }

        if (input2->requires_grad)
        {
            auto log_x = input1->log();
            auto temp = this->operator*(*log_x);
            input2->grad = grad->copy()->operator*(*temp);
        }
    }

    void Tensor::matmulBackward()
    {
        auto input1 = prev[0];
        auto input2 = prev[1];

        if (input1->requires_grad)
        {
            input1->grad = grad->copy()->matmul(*input2);
        }
        if (input2->requires_grad)
        {
            input2->grad = grad->copy()->matmul(*input1);
        }
    }
}