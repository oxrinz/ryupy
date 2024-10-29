#pragma once

#include "../../core/ITensor.h"
#include <stdexcept>
#include <iostream>

namespace ryupy {
    namespace cuda {
        class CudaTensor : public ITensor {
        public:
            CudaTensor(std::vector<int> size);
            CudaTensor* operator+(const ITensor &other) const;
            void printInfo() const override;
        };
    } 
} 