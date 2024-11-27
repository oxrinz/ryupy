#pragma once

#include "../../tensors/Tensor.h"
 
namespace ryupy
{
    namespace nn
    {
        class Layer {
            public:
                virtual ~Layer() = default;

                virtual std::shared_ptr<Tensor> forward(Tensor &tensor) = 0;
        };
    }
}