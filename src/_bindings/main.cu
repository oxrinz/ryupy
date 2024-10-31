#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "../core/Tensor.h"
#include "../backends/cpu/CpuTensor.h"
#include "../backends/cuda/CudaTensor.h"

namespace py = pybind11;

PYBIND11_MODULE(ryupy, m)
{
    py::class_<ryupy::Tensor>(m, "_Tensor");

    auto cpu = m.def_submodule("cpu");
    py::class_<ryupy::cpu::CpuTensor, ryupy::Tensor>(cpu, "Tensor")
        .def(py::init<py::object>())
        .def_property_readonly("shape", &ryupy::Tensor::getShape)
        .def_property_readonly("flattenedData", &ryupy::Tensor::getFlattenedData)
        .def_property_readonly("data", &ryupy::Tensor::getData);

    auto cuda = m.def_submodule("cuda");
    py::class_<ryupy::cuda::CudaTensor, ryupy::Tensor>(cuda, "Tensor")
        .def(py::init<py::object>())
        .def_property_readonly("shape", &ryupy::Tensor::getShape)
        .def_property_readonly("flattenedData", &ryupy::Tensor::getFlattenedData)
        .def_property_readonly("data", &ryupy::Tensor::getData);
}
