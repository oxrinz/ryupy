#include "../CudaTensor.h"
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace ryupy
{
    namespace cuda
    {
        std::shared_ptr<CudaTensor> CudaTensor::handleOperator(const CudaTensor &other, KernelFunc kernel) const
        {
            if (shape != other.shape)
            {
                throw std::invalid_argument("Tensors must be same shape");
            }

            auto result = std::make_shared<CudaTensor>(*this);

            cudaMalloc(&result->d_data, size);

            cudaMemcpy(result->d_data, this->d_data, size, cudaMemcpyDeviceToDevice);

            int threadsPerBlock = 256;
            int blocksPerGrid = (size / sizeof(float) + threadsPerBlock - 1) / threadsPerBlock;

            kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, other.d_data, result->d_data, size / sizeof(float));

            cudaDeviceSynchronize();

            return result;
        }

        std::shared_ptr<CudaTensor> CudaTensor::handleInPlaceOperator(const CudaTensor &other, KernelFunc kernel)
        {
            if (shape != other.shape)
            {
                throw std::invalid_argument("Tensors must be the same shape");
            }

            int threadsPerBlock = 256;
            int blocksPerGrid = (size / sizeof(float) + threadsPerBlock - 1) / threadsPerBlock;

            kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, other.d_data, d_data, size / sizeof(float));

            cudaDeviceSynchronize();

            return shared_from_this();
        }

        std::shared_ptr<CudaTensor> CudaTensor::handleShiftOperator(const int shift, KernelShiftFunc kernel) const
        {
            auto result = std::make_shared<CudaTensor>(*this);

            cudaMalloc(&result->d_data, size);

            cudaMemcpy(result->d_data, this->d_data, size, cudaMemcpyDeviceToDevice);

            int threadsPerBlock = 256;
            int blocksPerGrid = (size / sizeof(float) + threadsPerBlock - 1) / threadsPerBlock;

            kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, result->d_data, size / sizeof(float), shift);
            cudaDeviceSynchronize();

            return result;
        }

        std::shared_ptr<CudaTensor> CudaTensor::handleEmptyOperator(KernelEmptyFunc kernel) const
        {
            auto result = std::make_shared<CudaTensor>(*this);

            cudaMalloc(&result->d_data, size);

            cudaMemcpy(result->d_data, this->d_data, size, cudaMemcpyDeviceToDevice);

            int threadsPerBlock = 256;
            int blocksPerGrid = (size / sizeof(float) + threadsPerBlock - 1) / threadsPerBlock;

            kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, result->d_data, size / sizeof(float));
            cudaDeviceSynchronize();

            return result;
        }

        std::shared_ptr<CudaTensor> CudaTensor::handleInPlaceShiftOperator(int shift, KernelShiftFunc kernel)
        {
            int threadsPerBlock = 256;
            int blocksPerGrid = (size / sizeof(float) + threadsPerBlock - 1) / threadsPerBlock;

            // Apply the kernel directly to this tensor's data
            kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_data, size / sizeof(float), shift);
            cudaDeviceSynchronize();

            return shared_from_this();
        }

        std::shared_ptr<CudaTensor> CudaTensor::handleInPlaceEmptyOperator(KernelEmptyFunc kernel)
        {
            int threadsPerBlock = 256;
            int blocksPerGrid = (size / sizeof(float) + threadsPerBlock - 1) / threadsPerBlock;

            // Apply the kernel directly to this tensor's data
            kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_data, size / sizeof(float));
            cudaDeviceSynchronize();

            return shared_from_this();
        }
    }
}
