#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "../tensors/Tensor.h"

namespace py = pybind11;

inline void bind_layers(py::module &nn)
{
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
        .def("__getattr__", &ryupy::nn::LayerBank::getLayer)
        .def("zero_grad", &ryupy::nn::LayerBank::zero_grad);
}