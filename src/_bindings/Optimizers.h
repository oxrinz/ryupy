#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "../tensors/Tensor.h"

namespace py = pybind11;

inline void bind_optimizers(py::module &nn)
{
    auto optim = nn.def_submodule("optim");

    py::class_<ryupy::nn::optim::Optimizer, std::shared_ptr<ryupy::nn::optim::Optimizer>>(optim, "Optimizer")
        .def(py::init<std::shared_ptr<ryupy::nn::LayerBank>, float>())
        .def("step", &ryupy::nn::optim::Optimizer::step);

    using SGD = ryupy::nn::optim::SGD;
    py::class_<SGD, ryupy::nn::optim::Optimizer, std::shared_ptr<SGD>>(optim, "SGD")
        .def(py::init<std::shared_ptr<ryupy::nn::LayerBank>, float, float, float, float, bool>(),
             py::arg("layer_bank"),
             py::arg("lr") = 0.01f,
             py::arg("momentum") = 0.0f,
             py::arg("dampening") = 0.0f,
             py::arg("weight_decay") = 0.0f,
             py::arg("nesterov") = false)
        .def("step", &SGD::step);
}