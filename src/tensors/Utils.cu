#include "Tensor.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>

namespace ryupy
{
    std::pair<int, int> Tensor::calculate_fans(const std::vector<int> &shape)
    {
        if (shape.size() < 2)
        {
            return {1, 1};
        }

        int fan_in = shape[shape.size() - 1];
        int fan_out = shape[shape.size() - 2];

        if (shape.size() > 2)
        {
            for (size_t i = 0; i < shape.size() - 2; i++)
            {
                fan_in *= shape[i];
                fan_out *= shape[i];
            }
        }

        return {fan_in, fan_out};
    }
}
