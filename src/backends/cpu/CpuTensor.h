#pragma once

#include "../../core/ITensor.h"


namespace ryupy
{
    namespace cpu
    {
        class CpuTensor : public ITensor
        {
        public:
            CpuTensor(std::vector<int> size);

            CpuTensor* operator+(const ITensor &other) const;

            void printInfo() const override;
        };
    };
}
