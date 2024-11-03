#pragma once

#include "../../core/Tensor.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

namespace ryupy
{
    namespace cpu
    {
        class CpuTensor : public Tensor
        {
        public:
            std::vector<float> data;

            explicit CpuTensor(const py::object &data);
            py::object getData() const;
            py::object getFlattenedData() const;
        };
    }
}
