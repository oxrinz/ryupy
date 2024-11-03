#include "Tensor.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>

namespace ryupy
{
    Tensor::Tensor(const py::object &py_data)
    {
        shape = inferShape(py_data);
    }

    std::vector<int> Tensor::inferShape(const py::object &obj)
    {
        std::vector<int> shape;
        py::object current = obj.cast<py::list>();

        while (py::isinstance<py::list>(current) || py::isinstance<py::tuple>(current))
        {
            int current_size = py::len(current);
            shape.push_back(current_size);

            if (current_size == 0)
            {
                std::cout << "Encountered an empty list or tuple" << std::endl;
                break;
            }

            try
            {
                py::list current_list = current.cast<py::list>();
                py::object first_element = current_list[0];

                if (py::isinstance<py::list>(first_element) || py::isinstance<py::tuple>(first_element))
                {
                    current = first_element;
                }
                else
                {
                    break;
                }
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error accessing first element: " << e.what() << std::endl;
                break;
            }
        }

        return shape;
    }

    std::vector<float> Tensor::flattenData(const py::object &obj)
    {
        std::vector<float> flattened;

        if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj))
        {
            for (auto item : obj)
            {
                auto nested_flattened = flattenData(py::reinterpret_borrow<py::object>(item));       // Recursive call
                flattened.insert(flattened.end(), nested_flattened.begin(), nested_flattened.end()); // Append to result
            }
        }
        else if (py::isinstance<py::float_>(obj) || py::isinstance<py::int_>(obj))
        {
            flattened.push_back(obj.cast<float>());
        }
        else
        {
            throw std::invalid_argument("Unsupported data type encountered in the input object");
        }

        return flattened;
    }

    py::object Tensor::reshapeData(const std::vector<float> &data, const std::vector<int> &shape, int &index) const
    {
        int current_dim = shape[0];

        if (shape.size() == 1)
        {
            py::list result;
            for (int i = 0; i < current_dim; ++i)
            {
                result.append(data[index++]);
            }
            return result;
        }

        py::list result;
        std::vector<int> sub_shape(shape.begin() + 1, shape.end());
        for (int i = 0; i < current_dim; ++i)
        {
            result.append(reshapeData(data, sub_shape, index));
        }

        return result;
    }
}
