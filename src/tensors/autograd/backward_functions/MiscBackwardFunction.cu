#include "../../Tensor.h"
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../../operators/kernels/Kernels.h"
#include <iostream>
#include <cublas_v2.h>

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
        auto &a = prev[0];
        auto &b = prev[1];

        // Initialize gradients if needed
        if (a->requires_grad && !a->grad)
        {
            a->grad = zeros(a->shape);
        }
        if (b->requires_grad && !b->grad)
        {
            b->grad = zeros(b->shape);
        }

        // Handle matrix-matrix case first
        if (a->shape.size() == 2 && b->shape.size() == 2)
        {
            if (a->requires_grad)
            {
                auto grad_copy = grad->copy();
                auto b_transposed = b->transpose(0, 1);
                auto grad_a = grad_copy->matmul(*b_transposed);
                a->grad = a->grad->operator+(*grad_a);
            }
            if (b->requires_grad)
            {
                auto grad_copy = grad->copy();
                auto a_transposed = a->transpose(0, 1);
                auto grad_b = a_transposed->matmul(*grad_copy);
                b->grad = b->grad->operator+(*grad_b);
            }
            return;
        }

        // Handle matrix-vector case (2D * 1D)
        if (a->shape.size() == 2 && b->shape.size() == 1)
        {
            auto b_col = b->reshape({b->shape[0], 1});
            auto grad_col = grad->reshape({grad->shape[0], 1});
            if (a->requires_grad)
            {
                auto grad_copy = grad_col->copy();
                auto b_transposed = b_col->transpose(0, 1);
                auto grad_a = grad_copy->matmul(*b_transposed);
                a->grad = a->grad->operator+(*grad_a);
            }
            if (b->requires_grad)
            {
                auto grad_copy = grad_col->copy();
                auto a_transposed = a->transpose(0, 1);
                auto grad_b = a_transposed->matmul(*grad_copy);
                auto reshaped_grad_b = grad_b->reshape(b->shape);
                b->grad = b->grad->operator+(*reshaped_grad_b);
            }
            return;
        }

        // Handle vector-matrix case (1D * 2D)
        if (a->shape.size() == 1 && b->shape.size() == 2)
        {
            auto a_row = a->reshape({1, a->shape[0]});
            auto grad_row = grad->reshape({1, grad->shape[0]});
            if (a->requires_grad)
            {
                auto grad_copy = grad_row->copy();
                auto b_transposed = b->transpose(0, 1);
                auto grad_a = grad_copy->matmul(*b_transposed);
                auto reshaped_grad_a = grad_a->reshape(a->shape);
                a->grad = a->grad->operator+(*reshaped_grad_a);
            }
            if (b->requires_grad)
            {
                auto grad_copy = grad_row->copy();
                auto a_transposed = a_row->transpose(0, 1);
                auto grad_b = a_transposed->matmul(*grad_copy);
                b->grad = b->grad->operator+(*grad_b);
            }
            return;
        }

        // Handle batched cases with broadcasting
        if (a->requires_grad)
        {
            auto grad_copy = grad->copy();
            auto b_transposed = b->transpose(-2, -1);
            auto grad_a = grad_copy->matmul(*b_transposed);

            // Sum across broadcasted dimensions if necessary
            if (grad_a->shape != a->shape)
            {
                int dims_to_sum = grad_a->shape.size() - a->shape.size();
                for (int i = 0; i < dims_to_sum; i++)
                {
                    grad_a = grad_a->sum(0, true);
                }
                // Handle remaining dimensions that were broadcast
                for (size_t i = 0; i < a->shape.size(); i++)
                {
                    if (a->shape[i] == 1 && grad_a->shape[i] > 1)
                    {
                        grad_a = grad_a->sum(i, true);
                    }
                }
            }

            a->grad = a->grad->operator+(*grad_a);
        }

        if (b->requires_grad)
        {
            auto grad_copy = grad->copy();
            auto a_transposed = a->transpose(-2, -1);
            auto grad_b = a_transposed->matmul(*grad_copy);

            // Sum across broadcasted dimensions if necessary
            if (grad_b->shape != b->shape)
            {
                int dims_to_sum = grad_b->shape.size() - b->shape.size();
                for (int i = 0; i < dims_to_sum; i++)
                {
                    grad_b = grad_b->sum(0, true);
                }
                // Handle remaining dimensions that were broadcast
                for (size_t i = 0; i < b->shape.size(); i++)
                {
                    if (b->shape[i] == 1 && grad_b->shape[i] > 1)
                    {
                        grad_b = grad_b->sum(i, true);
                    }
                }
            }

            b->grad = b->grad->operator+(*grad_b);
        }
    }
}