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

namespace py = pybind11;

inline void bind_nn(py::module &m)
{
    auto nn = m.def_submodule("nn");

    bind_layers(nn);

    auto loss = nn.def_submodule("loss");

    loss.def("mse", &ryupy::nn::loss::mse_loss);

    bind_optimizers(nn);
}