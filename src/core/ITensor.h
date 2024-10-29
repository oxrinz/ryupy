#pragma once

#include <typeinfo>
#include <iostream>

namespace ryupy
{
    class Tensor;

    class ITensor
    {
    public:
        ITensor() : size(0) {}
        explicit ITensor(std::vector<int> s) : size(s) {}
        virtual ~ITensor() = default;

        std::vector<float> data;
        std::vector<int> shape;

        virtual void printInfo() const = 0;
    };
}