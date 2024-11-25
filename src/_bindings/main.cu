#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "../tensors/Tensor.h"
#include "../layers/LinearLayer.h"

namespace py = pybind11;

PYBIND11_MODULE(_ryupy, m)
{
    py::class_<ryupy::Tensor, std::shared_ptr<ryupy::Tensor>>(m, "Tensor")
        .def(py::init<py::object>())
        .def_property_readonly("shape", &ryupy::Tensor::getShape)
        .def_property_readonly("flattenedData", &ryupy::Tensor::getFlattenedData)
        .def_property_readonly("data", &ryupy::Tensor::getData)

        .def_property("grad", [](ryupy::Tensor &t)
                      { return t.grad; }, [](ryupy::Tensor &t, std::shared_ptr<ryupy::Tensor> new_grad)
                      { t.grad = new_grad; })
        .def_readwrite("requires_grad", &ryupy::Tensor::requires_grad)
        .def("backward", &ryupy::Tensor::backward)

        .def("__repr__", &ryupy::Tensor::repr)

        .def("__add__", &ryupy::Tensor::operator+)
        .def("__sub__", &ryupy::Tensor::operator-)
        .def("__mul__", &ryupy::Tensor::operator*)
        .def("__truediv__", &ryupy::Tensor::operator/)
        .def("__mod__", &ryupy::Tensor::operator%)

        .def("__iadd__", &ryupy::Tensor::operator+=)
        .def("__isub__", &ryupy::Tensor::operator-=)
        .def("__imul__", &ryupy::Tensor::operator*=)
        .def("__itruediv__", &ryupy::Tensor::operator/=)
        .def("__imod__", &ryupy::Tensor::operator%=)

        .def("__pow__", &ryupy::Tensor::pow)
        .def("__ipow__", &ryupy::Tensor::ipow)

        .def("__eq__", &ryupy::Tensor::operator==)
        .def("__ne__", &ryupy::Tensor::operator!=)
        .def("__lt__", &ryupy::Tensor::operator<)
        .def("__le__", &ryupy::Tensor::operator<=)
        .def("__gt__", &ryupy::Tensor::operator>)
        .def("__ge__", &ryupy::Tensor::operator>=)

        .def("__and__", &ryupy::Tensor::operator&)
        .def("__or__", &ryupy::Tensor::operator|)
        .def("__xor__", &ryupy::Tensor::operator^)
        .def("__invert__", &ryupy::Tensor::operator~)
        .def("__lshift__", &ryupy::Tensor::operator<<)
        .def("__rshift__", &ryupy::Tensor::operator>>)

        .def("__iand__", &ryupy::Tensor::operator&=)
        .def("__ior__", &ryupy::Tensor::operator|=)
        .def("__ixor__", &ryupy::Tensor::operator^=)
        .def("__ilshift__", &ryupy::Tensor::operator<<=)
        .def("__irshift__", &ryupy::Tensor::operator>>=)

        .def("__matmul__", &ryupy::Tensor::matmul);

    m.def("zeros", &ryupy::Tensor::zeros,
          py::arg("shape"),
          py::kw_only(),
          py::arg("grad") = false)
        .def("ones", &ryupy::Tensor::ones,
             py::arg("shape"),
             py::kw_only(),
             py::arg("grad") = false)
        .def("fill", &ryupy::Tensor::fill,
             py::arg("shape"),
             py::arg("value"),
             py::kw_only(),
             py::arg("grad") = false)
        .def("arange", &ryupy::Tensor::arange,
             py::arg("start"),
             py::arg("stop"),
             py::kw_only(),
             py::arg("step") = 1.0f,
             py::arg("grad") = false)
        .def("linspace", &ryupy::Tensor::linspace,
             py::arg("start"),
             py::arg("stop"),
             py::arg("num"),
             py::kw_only(),
             py::arg("grad") = false)
        .def("rand", &ryupy::Tensor::random_uniform,
             py::arg("shape"),
             py::kw_only(),
             py::arg("low") = 0.0f,
             py::arg("high") = 1.0f,
             py::arg("grad") = false)
        .def("randn", &ryupy::Tensor::random_normal,
             py::arg("shape"),
             py::kw_only(),
             py::arg("mean") = 0.0f,
             py::arg("std") = 1.0f,
             py::arg("grad") = false);

    auto nn = m.def_submodule("nn");

    using InitType = ryupy::nn::LinearLayer::InitType;
    py::enum_<InitType>(nn, "InitType")
        .value("XAVIER_UNIFORM", InitType::XAVIER_UNIFORM)
        .value("XAVIER_NORMAL", InitType::XAVIER_NORMAL)
        .value("KAIMING_UNIFORM", InitType::KAIMING_UNIFORM)
        .value("KAIMING_NORMAL", InitType::KAIMING_NORMAL)
        .export_values();

    py::class_<ryupy::nn::LinearLayer>(nn, "Linear")
        .def(py::init<int, int, InitType>())
        .def("forward", &ryupy::nn::LinearLayer::forward)
        .def_readwrite("weight", &ryupy::nn::LinearLayer::weight)
        .def_readwrite("bias", &ryupy::nn::LinearLayer::bias);
}