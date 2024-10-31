#include "CpuTensor.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace ryupy
{
    namespace cpu
    {
        CpuTensor::CpuTensor(const py::object &py_data) : Tensor(py_data)
        {
            data = flattenData(py_data);
        }

        py::object CpuTensor::getData() const
        {
            int index = 0;
            return reshapeData(data, shape, index);
        }

        py::object CpuTensor::getFlattenedData() const
        {
            return py::cast(data);
        }
    }
}