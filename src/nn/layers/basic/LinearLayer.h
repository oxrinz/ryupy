#pragma once

#include "../../../tensors/Tensor.h"
#include "../Layer.h"

namespace ryupy
{
    namespace nn
    {
        class LinearLayer : public Layer
        {
        public:
            enum class InitType
            {
                XAVIER_UNIFORM,
                XAVIER_NORMAL,
                KAIMING_UNIFORM,
                KAIMING_NORMAL
            };

            std::shared_ptr<Tensor> bias;

            LinearLayer(int in_features, int out_features, InitType init_type);
            std::shared_ptr<Tensor> forward(Tensor &tensor) override;

            static std::shared_ptr<LinearLayer> create(int in_features, int out_features, InitType init_type = InitType::XAVIER_UNIFORM);
        };
    }
}