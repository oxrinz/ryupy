#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "../core/Tensor.h"
#include "../backends/cuda/tensors/CudaTensor.h"
#include "../backends/cuda/layers/LinearLayer.h"

namespace py = pybind11;

PYBIND11_MODULE(ryupy, m)
{
    py::class_<ryupy::Tensor, std::shared_ptr<ryupy::Tensor>>(m, "_Tensor");

    auto cuda = m.def_submodule("cuda")
                    .def("zeros", &ryupy::cuda::CudaTensor::zeros)
                    .def("ones", &ryupy::cuda::CudaTensor::ones)
                    .def("fill", &ryupy::cuda::CudaTensor::fill)
                    .def("arange", &ryupy::cuda::CudaTensor::arange)
                    .def("linspace", &ryupy::cuda::CudaTensor::linspace)
                    .def("eye", &ryupy::cuda::CudaTensor::eye)
                    .def("rand", &ryupy::cuda::CudaTensor::random_uniform,
                         py::arg("shape"),
                         py::arg("low") = 0.0f,
                         py::arg("high") = 1.0f)
                    .def("randn", &ryupy::cuda::CudaTensor::random_normal,
                         py::arg("shape"),
                         py::arg("mean") = 0.0f,
                         py::arg("std") = 1.0f);

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

    auto cudann = cuda.def_submodule("nn");

    using InitType = ryupy::cuda::nn::LinearLayer::InitType;
    py::enum_<InitType>(cudann, "InitType")
        .value("XAVIER_UNIFORM", InitType::XAVIER_UNIFORM)
        .value("XAVIER_NORMAL", InitType::XAVIER_NORMAL)
        .value("KAIMING_UNIFORM", InitType::KAIMING_UNIFORM)
        .value("KAIMING_NORMAL", InitType::KAIMING_NORMAL)
        .export_values();

    py::class_<ryupy::cuda::nn::LinearLayer>(cudann, "Linear")
        .def(py::init<int, int, InitType>())
        .def("forward", &ryupy::cuda::nn::LinearLayer::forward)
        .def_readwrite("weight", &ryupy::cuda::nn::LinearLayer::weight)
        .def_readwrite("bias", &ryupy::cuda::nn::LinearLayer::bias);
}
