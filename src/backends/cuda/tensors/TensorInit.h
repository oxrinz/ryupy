#pragma once

#include "CudaTensor.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <memory>
#include <cudnn.h>

namespace py = pybind11;

namespace ryupy
{
    namespace cuda
    {
        CudaTensor rand(std::vector<int> shape)
        {
        }
    }
}
