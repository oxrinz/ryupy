#pragma once

#include "../layerbank/LayerBank.h"
#include "../layerbank/LayerBank.h"

namespace ryupy
{
    namespace nn
    {
        namespace optim
        {
            class Optimizer
            {
            protected:
                float lr;
                std::shared_ptr<LayerBank> layer_bank;
                std::unordered_map<std::shared_ptr<Tensor>, std::unordered_map<std::string, std::shared_ptr<Tensor>>> state;

            public:
                explicit Optimizer(std::shared_ptr<LayerBank> bank, float learning_rate = 0.01f)
                    : layer_bank(bank), lr(learning_rate) {}

                ~Optimizer() = default;

                virtual void step()
                {
                }
            };

            class SGD : public Optimizer
            {
            private:
                float momentum;
                float dampening;
                float weight_decay;
                bool nesterov;

            public:
                SGD(std::shared_ptr<LayerBank> bank, float lr = 0.01, float momentum = 0.0, float dampening = 0.0, float weight_decay = 0.0, bool nesterov = false);

                void step() override;
            };
        }
    }
}