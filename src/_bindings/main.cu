#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>  
#include "../core/ITensor.h"
#include "../backends/cpu/CpuTensor.h"
#include "../backends/cuda/CudaTensor.h"

namespace py = pybind11;

PYBIND11_MODULE(ryupy, m)
{
    // Expose ITensor to Python
    py::class_<ryupy::ITensor>(m, "_ITensor");

    // Expose CpuTensor with a constructor that takes 'size'
    auto cpu = m.def_submodule("cpu");
    py::class_<ryupy::cpu::CpuTensor, ryupy::ITensor>(cpu, "Tensor")
        .def(py::init<int>())  // Constructor that takes 'size'
        .def("print_info", &ryupy::cpu::CpuTensor::printInfo)
        .def(py::self + py::self)  // Bind operator+ as __add__
        .def_property_readonly("size", [](const ryupy::cpu::CpuTensor &t) { return t.size; })
        .def_property_readonly("data", [](const ryupy::cpu::CpuTensor &t) -> py::list {
            // Manually convert data to a Python list of floats
            py::list result;
            for (int i = 0; i < t.size; ++i) {
                result.append(t.data[i]);
            }
            return result;
        });

    // Expose CudaTensor with a constructor that takes 'size'
    auto cuda = m.def_submodule("cuda");
    py::class_<ryupy::cuda::CudaTensor, ryupy::ITensor>(cuda, "Tensor")
        .def(py::init<int>())  // Constructor that takes 'size'
        .def("print_info", &ryupy::cuda::CudaTensor::printInfo)
        .def(py::self + py::self)  // Bind operator+ as __add__
        .def_property_readonly("size", [](const ryupy::cuda::CudaTensor &t) { return t.size; })
        .def_property_readonly("data", [](const ryupy::cuda::CudaTensor &t) -> py::list {
            // Manually convert data to a Python list of floats
            py::list result;
            for (int i = 0; i < t.size; ++i) {
                result.append(t.data[i]);
            }
            return result;
        });
}
