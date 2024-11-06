#pragma once

#include <typeinfo>
#include <iostream>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

namespace ryupy
{
    class Tensor
    {
    public:
        explicit Tensor() : shape() {}
        explicit Tensor(const py::object &data);
        virtual ~Tensor() = default;

        std::vector<int> shape;

        const std::vector<int> &getShape() const
        {
            return shape;
        }

        virtual py::object getData() const = 0;
        virtual py::object getFlattenedData() const = 0;

    protected:
        std::vector<int>
        inferShape(const py::object &obj);
        std::vector<float> flattenPythonData(const py::object &obj);
        py::object reshapeData(const std::vector<float> &data, const std::vector<int> &shape) const;
    };
}