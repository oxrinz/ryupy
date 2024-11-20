#pragma once

#include "../../../core/Tensor.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <memory>
#include <cudnn.h>

namespace py = pybind11;

namespace ryupy
{
    namespace cuda
    {
        class CudaTensor : public Tensor, public std::enable_shared_from_this<CudaTensor>
        {
        public:
            float *d_data;
            int size;
            cudnnTensorDescriptor_t tensor_desc;

            CudaTensor() = default;
            explicit CudaTensor(const py::object &data);
            explicit CudaTensor(std::vector<int> shape);
            explicit CudaTensor(int size, std::vector<int> shape);
            virtual ~CudaTensor();

            static std::shared_ptr<CudaTensor> zeros(const std::vector<int> &shape);
            static std::shared_ptr<CudaTensor> ones(const std::vector<int> &shape);
            static std::shared_ptr<CudaTensor> fill(const std::vector<int> &shape, float value);
            static std::shared_ptr<CudaTensor> arange(float start, float stop, float step = 1.0f);
            static std::shared_ptr<CudaTensor> linspace(float start, float stop, int num);
            static std::shared_ptr<CudaTensor> eye(int n);
            static std::shared_ptr<CudaTensor> random_uniform(const std::vector<int> &shape, float low = 0.0f, float high = 1.0f);
            static std::shared_ptr<CudaTensor> random_normal(const std::vector<int> &shape, float mean = 0.0f, float std = 1.0f);

            py::object getData() const;
            py::object getFlattenedData() const;

            typedef void (*KernelFunc)(const float *, const float *, float *, int);
            typedef void (*KernelShiftFunc)(const float *, float *, int, int);
            typedef void (*KernelEmptyFunc)(const float *, float *, int);

            std::shared_ptr<CudaTensor> handleOperator(const CudaTensor &other, KernelFunc kernel) const;
            std::shared_ptr<CudaTensor> handleInPlaceOperator(const CudaTensor &other, KernelFunc kernel);
            std::shared_ptr<CudaTensor> handleShiftOperator(const int shift, KernelShiftFunc kernel) const;
            std::shared_ptr<CudaTensor> handleEmptyOperator(KernelEmptyFunc kernel) const;
            std::shared_ptr<CudaTensor> handleInPlaceShiftOperator(const int shift, KernelShiftFunc kernel);
            std::shared_ptr<CudaTensor> handleInPlaceEmptyOperator(KernelEmptyFunc kernel);

            // Basic arithmetic operators
            std::shared_ptr<CudaTensor> operator+(const CudaTensor &other) const;
            std::shared_ptr<CudaTensor> operator-(const CudaTensor &other) const;
            std::shared_ptr<CudaTensor> operator*(const CudaTensor &other) const;
            std::shared_ptr<CudaTensor> operator/(const CudaTensor &other) const;
            std::shared_ptr<CudaTensor> operator%(const CudaTensor &other) const;

            // In-place arithmetic operators
            std::shared_ptr<CudaTensor> operator+=(const CudaTensor &other);
            std::shared_ptr<CudaTensor> operator-=(const CudaTensor &other);
            std::shared_ptr<CudaTensor> operator*=(const CudaTensor &other);
            std::shared_ptr<CudaTensor> operator/=(const CudaTensor &other);
            std::shared_ptr<CudaTensor> operator%=(const CudaTensor &other);

            // Power operator
            std::shared_ptr<CudaTensor> pow(const CudaTensor &other) const;
            std::shared_ptr<CudaTensor> ipow(const CudaTensor &other);

            // Comparison operators
            std::shared_ptr<CudaTensor> operator==(const CudaTensor &other) const;
            std::shared_ptr<CudaTensor> operator!=(const CudaTensor &other) const;
            std::shared_ptr<CudaTensor> operator<(const CudaTensor &other) const;
            std::shared_ptr<CudaTensor> operator<=(const CudaTensor &other) const;
            std::shared_ptr<CudaTensor> operator>(const CudaTensor &other) const;
            std::shared_ptr<CudaTensor> operator>=(const CudaTensor &other) const;

            // Bitwise operators
            std::shared_ptr<CudaTensor> operator&(const CudaTensor &other) const;
            std::shared_ptr<CudaTensor> operator|(const CudaTensor &other) const;
            std::shared_ptr<CudaTensor> operator^(const CudaTensor &other) const;
            std::shared_ptr<CudaTensor> operator~() const;
            std::shared_ptr<CudaTensor> operator<<(int shift) const;
            std::shared_ptr<CudaTensor> operator>>(int shift) const;

            // In-place bitwise operators
            std::shared_ptr<CudaTensor> operator&=(const CudaTensor &other);
            std::shared_ptr<CudaTensor> operator|=(const CudaTensor &other);
            std::shared_ptr<CudaTensor> operator^=(const CudaTensor &other);
            std::shared_ptr<CudaTensor> operator<<=(int shift);
            std::shared_ptr<CudaTensor> operator>>=(int shift);

            // Matrix multiplication
            std::shared_ptr<CudaTensor> matmul(const CudaTensor &other) const; // Matrix multiplication

            // Logical operators (for boolean tensors)
            // std::shared_ptr<CudaTensor> logical_and(const CudaTensor &other) const;
            // std::shared_ptr<CudaTensor> logical_or(const CudaTensor &other) const;
            // std::shared_ptr<CudaTensor> logical_xor(const CudaTensor &other) const;
            // std::shared_ptr<CudaTensor> logical_not() const;

            // Unary operations
            // std::shared_ptr<CudaTensor> abs() const;  // Absolute value
            // std::shared_ptr<CudaTensor> sqrt() const; // Square root
            // std::shared_ptr<CudaTensor> exp() const;  // Exponential
            // std::shared_ptr<CudaTensor> log() const;  // Natural logarithm
            // std::shared_ptr<CudaTensor> sin() const;  // Sine
            // std::shared_ptr<CudaTensor> cos() const;  // Cosine
            // std::shared_ptr<CudaTensor> tan() const;  // Tangent

            // Floor, ceil, and rounding
            // std::shared_ptr<CudaTensor> floor() const;
            // std::shared_ptr<CudaTensor> ceil() const;
            // std::shared_ptr<CudaTensor> round() const;
        };
    }
}
