#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "../tensors/Tensor.h"
#include "../nn/layers/Layer.h"
#include "../nn/layers/basic/LinearLayer.h"
#include "../nn/layerbank/LayerBank.h"
#include "../nn/net/Net.h"
#include "../Ryu.h"
#include "../nn/loss/Loss.h"
#include "../nn/optim/Optim.h"

namespace py = pybind11;

PYBIND11_MODULE(_ryupy, m)
{
     m.def("ryu", &ryupy::print_ryu);

     py::class_<ryupy::Tensor, std::shared_ptr<ryupy::Tensor>>(m, "Tensor")
         .def_property_readonly("shape", &ryupy::Tensor::getShape)
         .def_property_readonly("flattenedData", &ryupy::Tensor::getFlattenedData)
         .def_property_readonly("data", &ryupy::Tensor::getData)

         .def_property("grad", [](ryupy::Tensor &t)
                       { return t.grad; }, [](ryupy::Tensor &t, std::shared_ptr<ryupy::Tensor> new_grad)
                       { t.grad = new_grad; })
         .def_readwrite("requires_grad", &ryupy::Tensor::requires_grad)
         .def("backward", &ryupy::Tensor::backward, py::arg("gradient") = nullptr)

         .def("copy", &ryupy::Tensor::copy)

         .def("__repr__", &ryupy::Tensor::repr)

         .def("__getitem__", &ryupy::Tensor::getItem)
         .def("__setitem__", &ryupy::Tensor::setItem)

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

         .def("sum", &ryupy::Tensor::sum)
         .def("__neg__", &ryupy::Tensor::negate)

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

     py::class_<ryupy::nn::Layer, std::shared_ptr<ryupy::nn::Layer>>(nn, "Layer")
         .def("forward", &ryupy::nn::Layer::forward)
         .def("__call__", &ryupy::nn::Layer::forward);

     py::class_<ryupy::nn::LinearLayer, ryupy::nn::Layer, std::shared_ptr<ryupy::nn::LinearLayer>>(nn, "Linear")
         .def(py::init(&ryupy::nn::LinearLayer::create))
         .def_readwrite("weight", &ryupy::nn::LinearLayer::weight)
         .def_readwrite("bias", &ryupy::nn::LinearLayer::bias);

     py::class_<ryupy::nn::LayerBank, std::shared_ptr<ryupy::nn::LayerBank>>(nn, "LayerBank")
         .def(py::init(&ryupy::nn::LayerBank::create))
         .def("__setattr__", &ryupy::nn::LayerBank::setLayer)
         .def("__getattr__", &ryupy::nn::LayerBank::getLayer);

     py::class_<ryupy::nn::Net, std::shared_ptr<ryupy::nn::Net>>(nn, "Net")
         .def(py::init(&ryupy::nn::Net::create))
         .def("__call__", &ryupy::nn::Net::forward);

     auto loss = nn.def_submodule("loss");

     loss.def("mse", &ryupy::nn::loss::mse_loss);

     auto optim = nn.def_submodule("optim");

     py::class_<ryupy::nn::optim::Optimizer, std::shared_ptr<ryupy::nn::optim::Optimizer>>(optim, "Optimizer")
         .def(py::init<std::shared_ptr<ryupy::nn::LayerBank>, float>())
         .def("step", &ryupy::nn::optim::Optimizer::step);

     using SGD = ryupy::nn::optim::SGD;
     py::class_<SGD, ryupy::nn::optim::Optimizer, std::shared_ptr<SGD>>(optim, "SGD")
         .def(py::init<std::shared_ptr<ryupy::nn::Net>, float, float, float, float, bool>(),
              py::arg("layer_bank"),
              py::arg("lr") = 0.01f,
              py::arg("momentum") = 0.0f,
              py::arg("dampening") = 0.0f,
              py::arg("weight_decay") = 0.0f,
              py::arg("nesterov") = false)
         .def("step", &SGD::step);
}