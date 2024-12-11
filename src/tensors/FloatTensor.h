#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <memory>
#include <optional>
#include <cudnn.h>

namespace py = pybind11;

namespace ryupy
{
    class FloatTensor : public std::enable_shared_from_this<Tensor>
    {
    }
}