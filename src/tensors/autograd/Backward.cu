#include "../Tensor.h"
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>

namespace ryupy
{
    void Tensor::backward()
    {
        if (!requires_grad)
        {
            throw std::runtime_error("Tensor does not require gradients");
        }

        if (!backward_fn)
        {
            throw std::runtime_error("Tensor has no backward function");
        }

        if (!grad)
        {
            grad = Tensor::ones(shape);
        }

        if (is_leaf)
        {
            return;
        }

        backward_fn();

        for (const auto &prev : prev)
        {
            if (prev->is_leaf == false)
            {
                prev->backward();
            }
        }
    }
}