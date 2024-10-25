#include <pybind11/pybind11.h>
#include <iostream>

void register_math_submodule(pybind11::module& m);
void register_tensor_submodule(pybind11::module& m);

PYBIND11_MODULE(ryupycuda, m) {
    m.def_submodule("math", "Math operations submodule");
    m.def_submodule("tensor", "Tensor operations submodule");

        register_math_submodule(m);
    register_tensor_submodule(m);
}