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

        if (!grad)
        {
            throw std::runtime_error("Gradient is null for one or more input tensors. Is grad on?");
        }

        if (input1->requires_grad)
        {
            if (grad->shape.size() == 1 && input2->shape.size() == 1)
            {
                auto grad_2d = grad->copy();
                auto input2_2d = input2->copy();
                grad_2d->shape = {grad->shape[0], 1};
                input2_2d->shape = {1, input2->shape[0]};
                input1->grad = grad_2d->matmul(*input2_2d);
                input1->grad->shape = input1->shape;
            }
            else
            {
                if (input2->shape.size() == 1)
                {
                    auto grad_2d = grad->copy();
                    auto input2_2d = input2->copy();
                    grad_2d->shape = {grad->shape[0], 1};
                    input2_2d->shape = {input2->shape[0], 1};
                    auto transposed = input2_2d->transpose();
                    input1->grad = grad_2d->matmul(*transposed);
                }
                else
                {
                    auto transposed = input2->transpose();
                    input1->grad = grad->matmul(*transposed);
                }
            }
        }

        if (input2->requires_grad)
        {
            if (grad->shape.size() == 1 && input1->shape.size() == 1)
            {
                auto grad_2d = grad->copy();
                auto input1_2d = input1->copy();
                grad_2d->shape = {grad->shape[0], 1};
                input1_2d->shape = {1, input1->shape[0]};
                input2->grad = grad_2d->matmul(*input1_2d);
                input2->grad->shape = input2->shape;
            }
            else
            {
                input2->grad = grad->copy()->matmul(*input1);
            }
        }
    }
}