#include "Tensor.h"
#include "kernels/Kernels.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <stdexcept>

namespace ryupy
{
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

    std::shared_ptr<Tensor> Tensor::reshape(const std::vector<int> &new_shape)
    {
        int current_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

        int new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());

        if (current_size != new_size)
        {
            throw std::invalid_argument(
                "Cannot reshape tensor of size " + std::to_string(current_size) +
                " into shape with size " + std::to_string(new_size));
        }

        auto reshaped = copy();
        reshaped->shape = new_shape;

        return reshaped;
    }

    std::shared_ptr<Tensor> Tensor::transpose(int dim0, int dim1)
    {
        if (dim0 < 0)
            dim0 += shape.size();
        if (dim1 < 0)
            dim1 += shape.size();
        if (dim0 < 0 || dim0 >= shape.size() || dim1 < 0 || dim1 >= shape.size())
        {
            throw std::invalid_argument("Invalid dimensions for transpose");
        }

        // Create new tensor
        auto result = std::make_shared<Tensor>(shape);
        std::swap(result->shape[dim0], result->shape[dim1]);

        // Calculate strides on host
        std::vector<int> old_strides(shape.size());
        std::vector<int> new_strides(shape.size());
        old_strides.back() = 1;
        new_strides.back() = 1;

        for (int i = shape.size() - 2; i >= 0; i--)
        {
            old_strides[i] = old_strides[i + 1] * shape[i + 1];
            new_strides[i] = new_strides[i + 1] * result->shape[i + 1];
        }

        // Allocate and copy shape and strides to device
        int *d_shape, *d_old_strides, *d_new_strides;
        cudaMalloc(&d_shape, shape.size() * sizeof(int));
        cudaMalloc(&d_old_strides, shape.size() * sizeof(int));
        cudaMalloc(&d_new_strides, shape.size() * sizeof(int));

        cudaMemcpy(d_shape, shape.data(), shape.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_old_strides, old_strides.data(), shape.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_new_strides, new_strides.data(), shape.size() * sizeof(int), cudaMemcpyHostToDevice);

        // Launch kernel
        int block_size = 256;
        int num_blocks = (result->size / sizeof(float) + block_size - 1) / block_size;
        transposeKernel<<<num_blocks, block_size>>>(
            d_data, result->d_data, d_shape, d_old_strides, d_new_strides,
            result->size / sizeof(float), dim0, dim1, shape.size());

        // Cleanup temporary device memory
        cudaFree(d_shape);
        cudaFree(d_old_strides);
        cudaFree(d_new_strides);

        // Set up N-dimensional tensor descriptor
        cudnnCreateTensorDescriptor(&result->tensor_desc);

        // Calculate strides for cuDNN
        std::vector<int> cudnn_strides(result->shape.size());
        int stride = 1;
        for (int i = result->shape.size() - 1; i >= 0; --i)
        {
            cudnn_strides[i] = stride;
            stride *= result->shape[i];
        }

        cudnnSetTensorNdDescriptor(
            result->tensor_desc,
            CUDNN_DATA_FLOAT,
            result->shape.size(),
            result->shape.data(),
            cudnn_strides.data());

        if (requires_grad)
        {
            result->requires_grad = true;
            result->is_leaf = false;
            result->prev = {shared_from_this()}; // Only store this tensor since transpose is unary

            // Store dimensions for backward pass
            int final_dim0 = dim0;
            int final_dim1 = dim1;

            result->backward_fn = [result, final_dim0, final_dim1]()
            {
                // The gradient of transpose is just another transpose with the same dimensions
                if (result->prev[0]->grad == nullptr)
                {
                    result->prev[0]->grad = result->grad->transpose(final_dim0, final_dim1);
                }
                else
                {
                    // Accumulate gradients if they already exist
                    auto transposed_grad = result->grad->transpose(final_dim0, final_dim1);
                    result->prev[0]->grad = result->prev[0]->grad->operator+(*transposed_grad);
                }
            };
        }

        return result;
    }

    bool Tensor::is_broadcastable_to(const std::vector<int> &target_shape) const
    {
        if (target_shape.size() < shape.size())
        {
            return false;
        }

        // Check from right to left
        size_t offset = target_shape.size() - shape.size();
        for (size_t i = 0; i < shape.size(); i++)
        {
            int current_dim = shape[i];
            int target_dim = target_shape[i + offset];

            if (current_dim != target_dim && current_dim != 1)
            {
                return false;
            }
        }
        return true;
    }

    std::shared_ptr<Tensor> Tensor::broadcast_to(const std::vector<int> &target_shape) const
    {
        if (!is_broadcastable_to(target_shape))
        {
            throw std::runtime_error("Cannot broadcast tensor to target shape");
        }

        // Create new tensor with target shape
        auto result = std::make_shared<Tensor>(target_shape);

        // Ensure input data is synchronized
        cudaDeviceSynchronize();

        // Only need shapes for this simpler version
        int *d_input_shape, *d_output_shape;
        cudaMalloc(&d_input_shape, shape.size() * sizeof(int));
        cudaMalloc(&d_output_shape, target_shape.size() * sizeof(int));

        cudaMemcpy(d_input_shape, shape.data(), shape.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_output_shape, target_shape.data(), target_shape.size() * sizeof(int), cudaMemcpyHostToDevice);

        // Launch kernel
        int block_size = 256;
        int num_elements = result->size / sizeof(float);
        int num_blocks = (num_elements + block_size - 1) / block_size;

        broadcastKernel<<<num_blocks, block_size>>>(
            d_data, result->d_data,
            d_input_shape, d_output_shape,
            num_elements);

        // Synchronize and check for errors
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            throw std::runtime_error(std::string("Kernel execution failed: ") + cudaGetErrorString(err));
        }

        // Clean up
        cudaFree(d_input_shape);
        cudaFree(d_output_shape);

        return result;
    }

    int Tensor::calculateBroadcastOffset(int flat_idx, const std::vector<int> &dims, const std::vector<int> &broadcast_shape)
    {
        std::vector<int> idx(dims.size());
        int temp = flat_idx;

        // Convert flat index to multi-dimensional index
        for (int i = dims.size() - 1; i >= 0; i--)
        {
            idx[i] = temp % broadcast_shape[i];
            temp /= broadcast_shape[i];
        }

        // Calculate offset considering broadcasting
        int offset = 0;
        int stride = 1;
        for (int i = dims.size() - 1; i >= 0; i--)
        {
            offset += (dims[i] == 1 ? 0 : idx[i]) * stride;
            stride *= dims[i];
        }

        return offset;
    }
}
