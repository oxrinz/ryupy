#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "../core/Tensor.h"
#include "../backends/cpu/CpuTensor.h"
#include "../backends/cuda/CudaTensor.h"

namespace py = pybind11;

PYBIND11_MODULE(ryupy, m)
{
    py::class_<ryupy::Tensor, std::shared_ptr<ryupy::Tensor>>(m, "_Tensor");

    // auto cpu = m.def_submodule("cpu");
    // py::class_<ryupy::cpu::CpuTensor, ryupy::Tensor, std::shared_ptr<ryupy::cpu::CpuTensor>>(cpu, "Tensor")
    //     .def(py::init<py::object>())
    //     .def_property_readonly("shape", &ryupy::Tensor::getShape)
    //     .def_property_readonly("flattenedData", &ryupy::Tensor::getFlattenedData)
    //     .def_property_readonly("data", &ryupy::Tensor::getData);

    auto cuda = m.def_submodule("cuda");
    py::class_<ryupy::cuda::CudaTensor, ryupy::Tensor, std::shared_ptr<ryupy::cuda::CudaTensor>>(cuda, "Tensor")
        .def(py::init<py::object>())
        .def_property_readonly("shape", &ryupy::Tensor::getShape)
        .def_property_readonly("flattenedData", &ryupy::Tensor::getFlattenedData)
        .def_property_readonly("data", &ryupy::Tensor::getData)

        .def("__add__", &ryupy::cuda::CudaTensor::operator+)
        .def("__sub__", &ryupy::cuda::CudaTensor::operator-)
        .def("__mul__", &ryupy::cuda::CudaTensor::operator*)
        .def("__truediv__", &ryupy::cuda::CudaTensor::operator/)
        .def("__mod__", &ryupy::cuda::CudaTensor::operator%)

        .def("__iadd__", &ryupy::cuda::CudaTensor::operator+=)
        .def("__isub__", &ryupy::cuda::CudaTensor::operator-=)
        .def("__imul__", &ryupy::cuda::CudaTensor::operator*=)
        .def("__itruediv__", &ryupy::cuda::CudaTensor::operator/=)
        .def("__imod__", &ryupy::cuda::CudaTensor::operator%=)

        .def("__pow__", &ryupy::cuda::CudaTensor::pow)
        .def("__ipow__", &ryupy::cuda::CudaTensor::ipow)

        .def("__eq__", &ryupy::cuda::CudaTensor::operator==)
        .def("__ne__", &ryupy::cuda::CudaTensor::operator!=)
        .def("__lt__", &ryupy::cuda::CudaTensor::operator<)
        .def("__le__", &ryupy::cuda::CudaTensor::operator<=)
        .def("__gt__", &ryupy::cuda::CudaTensor::operator>)
        .def("__ge__", &ryupy::cuda::CudaTensor::operator>=)

        .def("__and__", &ryupy::cuda::CudaTensor::operator&)
        .def("__or__", &ryupy::cuda::CudaTensor::operator|)
        .def("__xor__", &ryupy::cuda::CudaTensor::operator^)
        .def("__invert__", &ryupy::cuda::CudaTensor::operator~)
        .def("__lshift__", &ryupy::cuda::CudaTensor::operator<<)
        .def("__rshift__", &ryupy::cuda::CudaTensor::operator>>)

        .def("__iand__", &ryupy::cuda::CudaTensor::operator&=)
        .def("__ior__", &ryupy::cuda::CudaTensor::operator|=)
        .def("__ixor__", &ryupy::cuda::CudaTensor::operator^=)
        .def("__ilshift__", &ryupy::cuda::CudaTensor::operator<<=)
        .def("__irshift__", &ryupy::cuda::CudaTensor::operator>>=)
        
        .def("__matmul__", &ryupy::cuda::CudaTensor::matmul);
}
