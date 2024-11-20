#include "Tensor.h"
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cudnn.h>
#include <numeric>
#include <sstream>
#include <iomanip>

#define RESET "\033[0m"
#define WHITE "\033[37m"
#define BLUE "\033[34m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define MAGENTA "\033[35m"
#define CYAN "\033[36m"
#define BRIGHT_WHITE "\033[97m"
#define BRIGHT_BLUE "\033[94m"
#define BRIGHT_RED "\033[91m"
#define BRIGHT_GREEN "\033[92m"
#define BRIGHT_YELLOW "\033[93m"
#define BRIGHT_MAGENTA "\033[95m"
#define BRIGHT_CYAN "\033[96m"

namespace ryupy
{
    py::object Tensor::getFlattenedData() const
    {
        std::vector<float> hostData(size / sizeof(float));
        cudaMemcpy(hostData.data(), d_data, size, cudaMemcpyDeviceToHost);
        return py::cast(hostData);
    }

    py::object Tensor::getData() const
    {
        std::vector<float> hostData(size / sizeof(float));
        cudaMemcpy(hostData.data(), d_data, size, cudaMemcpyDeviceToHost);
        return reshapeData(hostData, shape);
    }


    const std::vector<int> Tensor::getShape() const
    {
        return shape;
    }

    std::string Tensor::repr() const
    {
        std::stringstream ss;
        ss << RED << "shape" << std::endl
           << "[";

        for (size_t i = 0; i < shape.size(); i++)
        {
            ss << shape[i];
            if (i < shape.size() - 1)
                ss << ", ";
        }

        ss << "]";
        ss << std::endl;
        ss << MAGENTA << "data" << std::endl;

        int total_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        std::vector<float> h_data(total_elements);
        cudaMemcpy(h_data.data(), d_data, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

        if (shape.size() == 1)
        {
            ss << "[";
            const int max_preview = 6;
            if (total_elements <= max_preview)
            {
                for (int i = 0; i < total_elements; i++)
                {
                    ss << std::fixed << std::setprecision(4) << h_data[i];
                    if (i < total_elements - 1)
                        ss << ", ";
                }
            }
            else
            {
                for (int i = 0; i < 3; i++)
                {
                    ss << std::fixed << std::setprecision(4) << h_data[i] << ", ";
                }
                ss << "..., ";
                for (int i = total_elements - 3; i < total_elements; i++)
                {
                    ss << std::fixed << std::setprecision(4) << h_data[i];
                    if (i < total_elements - 1)
                        ss << ", ";
                }
            }
            ss << "]";
        }

        else if (shape.size() == 2)
        {
            ss << "[\n";
            int rows = shape[0];
            int cols = shape[1];
            for (int i = 0; i < rows; i++)
            {
                ss << " [";
                for (int j = 0; j < cols; j++)
                {
                    ss << std::fixed << std::setprecision(4) << h_data[i * cols + j];
                    if (j < cols - 1)
                        ss << ", ";
                }
                ss << "]";
                if (i < rows - 1)
                    ss << ",\n";
            }
            ss << "\n]";
        }

        else if (shape.size() == 3)
        {
            ss << "[\n";
            int dim1 = shape[0];
            int dim2 = shape[1];
            int dim3 = shape[2];
            for (int i = 0; i < dim1; i++)
            {
                ss << " [\n";
                for (int j = 0; j < dim2; j++)
                {
                    ss << "  [";
                    for (int k = 0; k < dim3; k++)
                    {
                        ss << std::fixed << std::setprecision(4)
                           << h_data[i * dim2 * dim3 + j * dim3 + k];
                        if (k < dim3 - 1)
                            ss << ", ";
                    }
                    ss << "]";
                    if (j < dim2 - 1)
                        ss << ",\n";
                }
                ss << "\n ]";
                if (i < dim1 - 1)
                    ss << ",\n";
            }
            ss << "\n]";
        }

        else
        {
            ss << "<tensor of size ";
            for (size_t i = 0; i < shape.size(); i++)
            {
                ss << shape[i];
                if (i < shape.size() - 1)
                    ss << "×";
            }
            ss << ">";
        }

        return ss.str();
    }
}
