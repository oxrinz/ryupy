#pragma once

#include "../tensors/Tensor.h"

namespace ryupy
{
    namespace nn
    {
        class LinearLayer
        {
        public:
            enum class InitType
            {
                XAVIER_UNIFORM,
                XAVIER_NORMAL,
                KAIMING_UNIFORM,
                KAIMING_NORMAL
            };

            std::shared_ptr<Tensor> weight;
            std::shared_ptr<Tensor> bias;

            LinearLayer(int in_features, int out_features, InitType init_type);
            std::shared_ptr<Tensor> forward(const Tensor &tensor);
        };
    }
}