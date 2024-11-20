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

    std::vector<float> Tensor::flattenPythonData(const py::object &obj)
    {
        std::vector<int> shape;
        {
            py::object current = obj;
            while (py::isinstance<py::list>(current) || py::isinstance<py::tuple>(current))
            {
                py::sequence seq = current.cast<py::sequence>();
                shape.push_back(seq.size());
                if (seq.size() == 0)
                    break;
                current = seq[0];
            }
        }

        int n_dims = shape.size();
        if (n_dims < 2)
        {
            throw std::invalid_argument("Tensor must have at least 2 dimensions for matrix operations.");
        }

        int batch_dims = n_dims - 2;

        std::vector<float> flattened;

        std::function<void(int, const py::object &)> traverseBatch;
        traverseBatch = [&](int dim, const py::object &current)
        {
            if (dim == batch_dims)
            {
                py::object matrix = current;
                py::sequence rows = matrix.cast<py::sequence>();
                int m = rows.size();
                if (m != shape[batch_dims])
                {
                    throw std::invalid_argument("Mismatch in matrix row size.");
                }
                if (m == 0)
                    return;
                py::sequence cols = rows[0].cast<py::sequence>();
                int n = cols.size();
                if (n != shape[batch_dims + 1])
                {
                    throw std::invalid_argument("Mismatch in matrix column size.");
                }

                for (int col = 0; col < n; ++col)
                {
                    for (int row = 0; row < m; ++row)
                    {
                        py::object element = rows[row].cast<py::sequence>()[col];
                        if (py::isinstance<py::float_>(element) || py::isinstance<py::int_>(element))
                        {
                            flattened.push_back(element.cast<float>());
                        }
                        else
                        {
                            throw std::invalid_argument("Non-numeric element encountered in the input object");
                        }
                    }
                }
            }
            else
            {
                py::sequence seq = current.cast<py::sequence>();
                for (int i = 0; i < seq.size(); ++i)
                {
                    traverseBatch(dim + 1, seq[i]);
                }
            }
        };

        traverseBatch(0, obj);

        return flattened;
    }

    py::object Tensor::reshapeData(const std::vector<float> &data, const std::vector<int> &shape) const
    {
        int total_size = 1;
        for (int dim : shape)
        {
            total_size *= dim;
        }
        if (data.size() != total_size)
        {
            throw std::invalid_argument("Data size does not match shape.");
        }

        int n_dims = shape.size();
        if (n_dims < 2)
        {
            py::list raw_list;
            for (float value : data)
            {
                raw_list.append(value);
            }
            return raw_list;
        }

        int batch_dims = n_dims - 2; 

        std::vector<int> batch_strides(batch_dims, 1);
        for (int i = batch_dims - 2; i >= 0; --i)
        {
            batch_strides[i] = batch_strides[i + 1] * shape[i + 1];
        }

        int m = shape[n_dims - 2];
        int n = shape[n_dims - 1];

        int matrix_size = m * n;

        std::function<py::object(int, int)> buildStructure;
        buildStructure = [&](int dim, int index) -> py::object
        {
            if (dim == batch_dims)
            {
                py::list matrix;
                for (int row = 0; row < m; ++row)
                {
                    py::list row_list;
                    for (int col = 0; col < n; ++col)
                    {
                        int linear_index = index + col * m + row;
                        float value = data[linear_index];
                        row_list.append(value);
                    }
                    matrix.append(row_list);
                }
                return matrix;
            }
            else
            {
                py::list batch_list;
                int stride = batch_strides[dim] * matrix_size;
                for (int i = 0; i < shape[dim]; ++i)
                {
                    batch_list.append(buildStructure(dim + 1, index + i * stride));
                }
                return batch_list;
            }
        };

        return buildStructure(0, 0);
    }

}
