#pragma once

#include "../../core/Tensor.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

namespace ryupy
{
    namespace cuda
    {
        class CudaTensor : public Tensor
        {
        public:
            float* d_data;
            int size;

            explicit CudaTensor(const py::object &data);
            virtual ~CudaTensor();  
            py::object getData() const;
            py::object getFlattenedData() const;
        };
    }
}
