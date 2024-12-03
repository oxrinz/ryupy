#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "../tensors/Tensor.h"
#include "../nn/layers/Layer.h"
#include "../nn/layers/basic/LinearLayer.h"
#include "../nn/layerbank/LayerBank.h"
#include "../Ryu.h"
#include "../nn/loss/Loss.h"
#include "../nn/optim/Optim.h"
#include "../data/Dataset.h" 
 
#include "Tensor.h"
#include "Layers.h" 
#include "Optimizers.h"
#include "NN.h" 

namespace py = pybind11; 

PYBIND11_MODULE(_ryupy, m)
{
    m.def("ryu", &ryupy::print_ryu);

    bind_tensor(m);

    bind_nn(m);

    py::class_<ryupy::Dataset, std::shared_ptr<ryupy::Dataset>>(m, "Dataset")
        .def(py::init(&ryupy::Dataset::create),
             py::arg("inputs"),
             py::arg("targets"))
        .def_readwrite("batch_size", &ryupy::Dataset::batch_size)
        .def_readwrite("shuffle", &ryupy::Dataset::shuffle)
        .def("reset", &ryupy::Dataset::reset)
        .def("has_next", &ryupy::Dataset::has_next)
        .def("next", &ryupy::Dataset::next)
        .def("__iter__", [](ryupy::Dataset &d)
             {
            d.reset();
            return d; })
        .def("__next__", [](ryupy::Dataset &d)
             {
            if (!d.has_next()) {
                throw py::stop_iteration();
            }
            return d.next(); });
}