#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <memory>
#include <optional>
#include <cudnn.h>

namespace py = pybind11;

namespace ryupy
{
    class Tensor : public std::enable_shared_from_this<Tensor>
    {
    public:
        std::vector<int> shape;
        float *d_data;
        // IMPORTANT!! Currently the size is BYTES, not the amount of elements. Change this later
        int size;
        cudnnTensorDescriptor_t tensor_desc;

        // Autograd
        std::vector<std::shared_ptr<Tensor>> prev;
        std::shared_ptr<Tensor> grad;
        bool requires_grad;
        std::function<void(void)> backward_fn;
        bool is_leaf;
        void backward(const Tensor *gradient = nullptr);

        // Constructors
        Tensor() = default;
        explicit Tensor(std::vector<int> shape);
        virtual ~Tensor();

        // Initializers
        static std::shared_ptr<Tensor> zeros(const std::vector<int> &shape, bool grad = false);
        static std::shared_ptr<Tensor> ones(const std::vector<int> &shape, bool grad = false);
        static std::shared_ptr<Tensor> fill(const std::vector<int> &shape, float value, bool grad = false);
        static std::shared_ptr<Tensor> arange(float start, float stop, float step = 1.0f, bool grad = false);
        static std::shared_ptr<Tensor> linspace(float start, float stop, int num, bool grad = false);
        static std::shared_ptr<Tensor> eye(int n, bool grad = false);
        static std::shared_ptr<Tensor> random_uniform(const std::vector<int> &shape, float low = 0.0f, float high = 1.0f, bool grad = false);
        static std::shared_ptr<Tensor> random_normal(const std::vector<int> &shape, float mean = 0.0f, float std = 1.0f, bool grad = false);
        static std::shared_ptr<Tensor> xavier_normal(const std::vector<int> &shape, bool grad = false);
        static std::shared_ptr<Tensor> xavier_uniform(const std::vector<int> &shape, bool grad = false);
        static std::shared_ptr<Tensor> kaiming_normal(const std::vector<int> &shape, bool grad = false);
        static std::shared_ptr<Tensor> kaiming_uniform(const std::vector<int> &shape, bool grad = false);

        // Utils
        std::vector<int> calculate_strides(const std::vector<int> &shape) const;
        static std::pair<int, int> calculate_fans(const std::vector<int> &shape);
        std::shared_ptr<Tensor> copy() const;

        // Shaping functions
        std::vector<int> inferShape(const py::object &obj);
        std::vector<float> flattenPythonData(const py::object &obj);
        py::object reshapeData(const std::vector<float> &data, const std::vector<int> &shape) const;
        std::shared_ptr<Tensor> reshape(const std::vector<int> &new_shape);
        std::shared_ptr<Tensor> transpose(int dim0, int dim1);

        // Broadcasting functionality
        bool is_broadcastable_to(const std::vector<int> &target_shape) const;
        std::shared_ptr<Tensor> broadcast_to(const std::vector<int> &target_shape) const;
        static int calculateBroadcastOffset(int flat_idx, const std::vector<int> &dims, const std::vector<int> &broadcast_shape);

        // Python interface shit
        std::shared_ptr<Tensor> parent; // Shit solution but it works, change later maybe
        int parent_index;

        std::string repr() const;
        const std::vector<int> getShape() const;
        py::object getData() const;
        py::object getFlattenedData() const;
        py::object getItem(int index);
        void setItem(int index, const py::object &value);

        // Operation kernel typedefs
        typedef void (*KernelFunc)(const float *, const float *, float *, int);
        typedef void (*KernelShiftFunc)(const float *, float *, int, int);
        typedef void (*KernelEmptyFunc)(const float *, float *, int);

        // Operation handlers
        std::shared_ptr<Tensor> handleOperator(Tensor &other, KernelFunc kernel, void (Tensor::*backward_function)() = nullptr);
        std::shared_ptr<Tensor> handleInPlaceOperator(Tensor &other, KernelFunc kernel);
        std::shared_ptr<Tensor> handleShiftOperator(const int shift, KernelShiftFunc kernel) const;
        std::shared_ptr<Tensor> handleEmptyOperator(KernelEmptyFunc kernel) const;
        std::shared_ptr<Tensor> handleInPlaceShiftOperator(const int shift, KernelShiftFunc kernel);
        std::shared_ptr<Tensor> handleInPlaceEmptyOperator(KernelEmptyFunc kernel);
        float handleReduceOperator(KernelEmptyFunc kernel) const;

        // Basic arithmetic operators
        std::shared_ptr<Tensor> operator+(Tensor &other);
        std::shared_ptr<Tensor> operator-(Tensor &other);
        std::shared_ptr<Tensor> operator*(Tensor &other);
        std::shared_ptr<Tensor> operator/(Tensor &other);
        std::shared_ptr<Tensor> operator%(Tensor &other);

        // In-place arithmetic operators
        std::shared_ptr<Tensor> operator+=(Tensor &other);
        std::shared_ptr<Tensor> operator-=(Tensor &other);
        std::shared_ptr<Tensor> operator*=(Tensor &other);
        std::shared_ptr<Tensor> operator/=(Tensor &other);
        std::shared_ptr<Tensor> operator%=(Tensor &other);

        // Power operator
        std::shared_ptr<Tensor> pow(Tensor &other);
        std::shared_ptr<Tensor> ipow(Tensor &other);

        // Comparison operators
        std::shared_ptr<Tensor> operator==(Tensor &other);
        std::shared_ptr<Tensor> operator!=(Tensor &other);
        std::shared_ptr<Tensor> operator<(Tensor &other);
        std::shared_ptr<Tensor> operator<=(Tensor &other);
        std::shared_ptr<Tensor> operator>(Tensor &other);
        std::shared_ptr<Tensor> operator>=(Tensor &other);

        // Bitwise operators
        std::shared_ptr<Tensor> operator&(Tensor &other);
        std::shared_ptr<Tensor> operator|(Tensor &other);
        std::shared_ptr<Tensor> operator^(Tensor &other);
        std::shared_ptr<Tensor> operator~() const;
        std::shared_ptr<Tensor> operator<<(int shift) const;
        std::shared_ptr<Tensor> operator>>(int shift) const;

        // In-place bitwise operators
        std::shared_ptr<Tensor> operator&=(Tensor &other);
        std::shared_ptr<Tensor> operator|=(Tensor &other);
        std::shared_ptr<Tensor> operator^=(Tensor &other);
        std::shared_ptr<Tensor> operator<<=(int shift);
        std::shared_ptr<Tensor> operator>>=(int shift);

        // Matrix multiplication
        std::shared_ptr<Tensor> matmul(Tensor &other);

        // Logical operators (for boolean tensors)
        // std::shared_ptr<Tensor> logical_and(const Tensor &other) const;
        // std::shared_ptr<Tensor> logical_or(const Tensor &other) const;
        // std::shared_ptr<Tensor> logical_xor(const Tensor &other) const;
        // std::shared_ptr<Tensor> logical_not() const;

        // Unary operations
        // std::shared_ptr<Tensor> abs() const;  // Absolute value
        // std::shared_ptr<Tensor> sqrt() const; // Square root
        // std::shared_ptr<Tensor> exp() const;  // Exponential
        std::shared_ptr<Tensor> log() const;
        // std::shared_ptr<Tensor> sin() const;  // Sine
        // std::shared_ptr<Tensor> cos() const;  // Cosine
        // std::shared_ptr<Tensor> tan() const;  // Tangent
        std::shared_ptr<Tensor> negate();
        std::shared_ptr<ryupy::Tensor> sum(const std::optional<int> &dim = std::nullopt, bool keepdim = false);

        // Floor, ceil, and rounding
        // std::shared_ptr<Tensor> floor() const;
        // std::shared_ptr<Tensor> ceil() const;
        // std::shared_ptr<Tensor> round() const;

        // Backward functions
        void addBackward();
        void subtractBackward();
        void multiplyBackward();
        void divideBackward();
        void powerBackward();

        // Implement later potentially
        // void addInPlaceBackward();
        // void subtractInPlaceBackward();
        // void multiplyInPlaceBackward();
        // void divideInPlaceBackward();
        // void powerInPlaceBackward();

        void matmulBackward();
    };
}
