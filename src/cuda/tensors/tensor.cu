#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdlib>
#include <vector>

class Tensor {
public:
    std::vector<float> vector;

    Tensor(const std::vector<float>& defaultVector) {
        vector = defaultVector;
    }
};

void register_tensor_submodule(pybind11::module& m) {
    pybind11::module tensor_submodule = m.def_submodule("tensor");
    pybind11::class_<Tensor>(tensor_submodule, "Tensor")
        .def(pybind11::init<const std::vector<float>&>())
        .def_readwrite("vector", &Tensor::vector);
}