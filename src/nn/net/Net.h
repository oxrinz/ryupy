#pragma once

#include <functional>
#include "../layerbank/LayerBank.h"

namespace ryupy
{
    namespace nn
    {
        class Net
        {
        public:
            using ForwardFunction = std::function<std::shared_ptr<Tensor>(std::shared_ptr<Tensor>)>;

            ForwardFunction m_forward_fn;
            std::shared_ptr<LayerBank> m_layer_bank;

            ~Net() = default;

            Net(std::shared_ptr<LayerBank> layer_bank, ForwardFunction forward_fn);

            static std::shared_ptr<Net> create(std::shared_ptr<LayerBank> layer_bank, ForwardFunction forward_fn);

            std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input);
        };
    }
}